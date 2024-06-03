# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from omegaconf import DictConfig
from torch import Tensor, nn
import torch
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_pos2global, torch_rad2local, torch_rad2global
from utils.pooling import seq_pooling
from utils.rpe import get_rel_pose, get_rel_dist, get_tgt_knn_idx
from utils.pose_emb import PoseEmb
from .modules.distributions import MyDist, DestCategorical, DiagGaussian
from .modules.mlp import MLP
from .modules.transformer_rpe import TransformerBlockRPE
from .modules.multi_agent_gru import MultiAgentGRULoop
from .modules.input_encoder import InputEncoder
from .modules.polyline_encoder import PolylineEncoder


class NaviEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: DictConfig,
        navi_mode: str,
        navi_dim: Optional[int],
        pairwise_relative: bool,
        dest_detach_mp_feature: bool,
        mp_pose_emb: PoseEmb,
        pose_rpe: PoseEmb,
    ) -> None:
        super().__init__()
        self.navi_mode = navi_mode
        self.pairwise_relative = pairwise_relative
        self.dest_detach_mp_feature = dest_detach_mp_feature
        if self.navi_mode == "dummy":
            self.require_update, self.dummy = False, True
        elif self.navi_mode == "dest":
            self.require_update, self.dummy = pairwise_relative, False
            self.mlp_mp = MLP([hidden_dim, hidden_dim], end_layer_activation=False)
            if pairwise_relative:
                self.pose_emb = pose_rpe
                self.mlp_pe = MLP([self.pose_emb.out_dim, hidden_dim], end_layer_activation=False)
        elif self.navi_mode == "goal":
            self.require_update, self.dummy = pairwise_relative, False
            self.pose_emb = pose_rpe if pairwise_relative else mp_pose_emb
            self.mlp = MLP([self.pose_emb.out_dim + 1, hidden_dim], end_layer_activation=False)
        elif self.navi_mode == "cmd":
            self.require_update, self.dummy = False, False
            self.mlp = MLP([navi_dim, hidden_dim], end_layer_activation=False)

    def forward(
        self, ag_navi: Optional[Tensor], ag_pose: Tensor, mp_token_feature: Tensor, mp_token_pose: Tensor
    ) -> Optional[Tensor]:
        """
        Args:
            ag_navi: 
                dest: [n_sc, n_ag], int64 index to map n_pl
                goal: [n_sc, n_ag, 4], (x,y,yaw,spd) in global coordinate
                cmd: [n_sc, n_ag, 8]
            ag_pose: [n_sc, n_ag, 3] (x,y,yaw) in global coordinate
            mp_token_feature: [n_sc, n_mp, hidden_dim]
            mp_token_pose: [n_sc, n_mp, 3]

        Return:
            navi_feature: [n_sc, n_ag, hidden_dim]
        """
        if self.navi_mode == "dest":
            if self.dest_detach_mp_feature:
                mp_token_feature = mp_token_feature.detach()
            _idx_sc = torch.arange(ag_navi.shape[0]).unsqueeze(1)  # [n_sc, 1]

            navi_feature = self.mlp_mp(mp_token_feature[_idx_sc, ag_navi])  # [n_sc, n_ag, hidden_dim]

            if self.pairwise_relative:
                global_pose = mp_token_pose[_idx_sc, ag_navi]  # [n_sc, n_ag, 3]
                # [n_sc, n_ag, 2]
                xy = torch_pos2local(
                    global_pose[:, :, None, :2], ag_pose[:, :, None, :2], torch_rad2rot(ag_pose[:, :, -1])
                ).squeeze(2)
                yaw = torch_rad2local(global_pose[:, :, 2:3], ag_pose[:, :, -1], cast=False)  # [n_sc, n_ag, 1]
                navi_feature = navi_feature + self.mlp_pe(self.pose_emb(xy, yaw))

        elif self.navi_mode == "goal":
            # goal: [n_sc, n_ag, 4], (x,y,yaw,spd) in global coordinate
            with torch.no_grad():
                xy, yaw, spd = ag_navi[:, :, :2], ag_navi[:, :, 2:3], ag_navi[:, :, 3:4]  # [n_sc, n_ag, 2/1/1]
                if self.pairwise_relative:
                    # [n_sc, n_ag, 2]
                    xy = torch_pos2local(
                        xy.unsqueeze(2), ag_pose[:, :, None, :2], torch_rad2rot(ag_pose[:, :, -1])
                    ).squeeze(2)
                    yaw = torch_rad2local(yaw, ag_pose[:, :, -1], cast=False)  # [n_sc, n_ag, 1]
            navi_feature = self.mlp(torch.cat([self.pose_emb(xy, yaw), spd], dim=-1))

        elif self.navi_mode == "cmd":
            navi_feature = self.mlp(ag_navi.type_as(ag_pose))
        elif self.navi_mode == "dummy":
            navi_feature = None
        else:
            raise NotImplementedError

        return navi_feature


class NaviPredictor(nn.Module):
    def __init__(
        self,
        navi_mode: str,
        detach_input: bool,
        rnn_res_add: bool,
        n_layer_tf: int,
        n_layer_mlp: int,
        navi_dim: Optional[int],
        mlp_use_layernorm: bool,
        k_tgt_knn: float,
        k_dist_limit: float,
        ag_encoder: DictConfig,
        goal_log_std: float,
        pose_rpe: PoseEmb,
    ) -> None:
        super().__init__()
        self.navi_mode = navi_mode
        self.detach_input = detach_input
        self.rnn_res_add = rnn_res_add
        self.pose_rpe = pose_rpe

        self.pairwise_relative = ag_encoder["pairwise_relative"]
        self.temp_window_size = ag_encoder["temp_window_size"]
        hidden_dim = ag_encoder["hidden_dim"]
        input_encoder = ag_encoder["input_encoder"]
        temp_encoder = ag_encoder["temp_encoder"]

        if self.temp_window_size <= 0 and self.pairwise_relative:  # pairwise-relative RNN
            input_pe_dim = 0  # no global/local pose, just the difference
            self.pose_emb = None
        else:
            pe_dim = hidden_dim if input_encoder.mode == "add" else hidden_dim // 2
            self.pose_emb = PoseEmb(pe_dim=pe_dim, **ag_encoder["pose_emb"])
            input_pe_dim = self.pose_emb.out_dim

        attr_dim = ag_encoder["ag_attr_dim"] + ag_encoder["ag_motion_dim"]
        if self.temp_window_size > 0:
            self.register_buffer("hist_ohe", torch.eye(self.temp_window_size))
            attr_dim += self.temp_window_size
            self.temp_encoder = PolylineEncoder(hidden_dim=hidden_dim, tf_cfg=ag_encoder["tf_cfg"], **temp_encoder)
        else:
            self.temp_encoder = MultiAgentGRULoop(hidden_dim, temp_encoder["n_layer"], temp_encoder["mlp_dropout_p"])
            self.rnn_temp_pool_mode = ag_encoder["rnn_latent_temp_pool_mode"]

        self.input_encoder = InputEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim, pe_dim=input_pe_dim, **input_encoder
        )

        if self.navi_mode == "dest":
            mlp_in_dim = 2 * hidden_dim
            if self.pairwise_relative:
                mlp_in_dim += self.pose_rpe.out_dim
            self.mlp = MLP(
                [mlp_in_dim] + [hidden_dim] * (n_layer_mlp - 1) + [1],
                end_layer_activation=False,
                use_layernorm=mlp_use_layernorm,
            )
        else:
            self.n_tgt_knn = int(ag_encoder["n_tgt_knn"] * k_tgt_knn)
            self.dist_limit = ag_encoder["dist_limit"] * k_dist_limit
            d_rpe = self.pose_rpe.out_dim if self.pairwise_relative else -1
            self.tf_ag2mp = TransformerBlockRPE(
                n_layer=n_layer_tf, mode="enc_cross_attn", d_rpe=d_rpe, **ag_encoder["tf_cfg"]
            )

            self.mlp = MLP(
                [hidden_dim] * n_layer_mlp + [navi_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
            )
            if self.navi_mode == "goal":
                self.log_std = nn.Parameter(goal_log_std * torch.ones(navi_dim), requires_grad=True)

    def forward(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag, n_step]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_motion: Tensor,  # [n_sc, n_ag, n_step, ag_motion_dim]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3], (x,y,yaw)
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        ag_type: Tensor,  # [n_sc, n_ag, 3] [Vehicle=0, Pedestrian=1, Cyclist=2] one hot
        mp_token_type: Tensor,  # [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11
    ) -> Optional[MyDist]:
        if self.navi_mode == "dummy":
            return None
        if self.detach_input:
            ag_motion = ag_motion.detach()
            ag_pose = ag_pose.detach()
            mp_token_feature = mp_token_feature.detach()

        # ! get ag_token_feature: [n_sc, n_ag, hidden_dim]
        n_sc, n_ag, n_step = ag_valid.shape
        ag_token_valid = ag_valid.any(-1)
        ag_invalid, ag_token_invalid = ~ag_valid, ~ag_token_valid

        if self.pairwise_relative:
            ag_token_pose = seq_pooling(ag_pose, ag_invalid, "last_valid", ag_valid)  # [n_sc, n_ag, 3]
            ref_pos = ag_token_pose[:, :, None, :2]  # [n_sc, n_ag, 1, 2]
            ref_yaw = ag_token_pose[..., -1]  # [n_sc, n_ag]
            ref_rot = torch_rad2rot(ref_yaw)  # [n_sc, n_ag, 2, 2]

        if self.temp_window_size > 0:  # HPTR VectorNet
            if n_step > self.temp_window_size:
                ag_pose = ag_pose[:, :, -self.temp_window_size :]
                ag_motion = ag_motion[:, :, -self.temp_window_size :]
                ag_invalid = ag_invalid[:, :, -self.temp_window_size :]
                n_step = self.temp_window_size

            ag_xy, ag_yaw = ag_pose[..., :2], ag_pose[..., 2:3]  # [n_sc, n_ag, n_step, 2/1]
            if self.pairwise_relative:
                ag_xy = torch_pos2local(ag_xy, ref_pos, ref_rot)
                ag_yaw = torch_rad2local(ag_yaw.squeeze(-1), ref_yaw, cast=False).unsqueeze(-1)

            ag_attr = torch.cat(
                [
                    ag_attr.unsqueeze(2).expand(-1, -1, n_step, -1),
                    ag_motion,
                    self.hist_ohe[None, None, -n_step:, :].expand(n_sc, n_ag, -1, -1),
                ],
                dim=-1,
            )

            ag_feature = self.input_encoder(ag_attr, self.pose_emb(ag_xy, ag_yaw))  # [n_sc, n_ag, n_step, hidden_dim]

            ag_token_feature = self.temp_encoder(ag_feature, ag_invalid)  # [n_sc, n_ag, hidden_dim]

        else:  # TrafficBots RNN
            if self.pairwise_relative:
                ag_pose_emb = None
            else:
                ag_pose_emb = self.pose_emb(ag_pose[..., :2], ag_pose[..., 2:3])  # [n_sc, n_ag, n_step, 2/1]

            ag_feature = self.input_encoder(
                torch.cat([ag_attr.unsqueeze(2).expand(-1, -1, n_step, -1), ag_motion], dim=-1), ag_pose_emb
            )  # [n_sc, n_ag, n_step, hidden_dim]

            ag_token_feature, _ = self.temp_encoder(ag_feature, ag_invalid)
            if self.rnn_res_add:
                ag_token_feature = ag_token_feature + ag_feature
            ag_token_feature = seq_pooling(ag_token_feature, ag_invalid, self.rnn_temp_pool_mode, ag_valid)

        # ! predict navi output from ag_token_feature: [n_sc, n_ag, hidden_dim]
        if self.navi_mode == "dest":
            ag_mp_feature = torch.cat(
                [
                    ag_token_feature.unsqueeze(2).expand(-1, -1, mp_token_invalid.shape[1], -1),
                    mp_token_feature.unsqueeze(1).expand(-1, n_ag, -1, -1),
                ],
                dim=-1,
            )  # [n_sc, n_ag, n_mp, hidden_dim*2]

            if self.pairwise_relative:
                # rpe_ag2mp: [n_sc, n_ag, n_mp, 3] (x,y,yaw)
                rpe_ag2mp, _ = get_rel_pose(ag_token_pose, ag_token_invalid, mp_token_pose, mp_token_invalid)
                rpe_ag2mp = self.pose_rpe(xy=rpe_ag2mp[..., :2], dir=rpe_ag2mp[..., [2]])
                ag_mp_feature = torch.cat([ag_mp_feature, rpe_ag2mp], dim=-1)

            logits = self.mlp(ag_mp_feature).squeeze(-1)  # [n_sc, n_ag, n_mp]

            # ! mask logits using agent type and map type
            # WOMD: FREEWAY = 0, SURFACE_STREET = 1, STOP_SIGN = 2, BIKE_LANE = 3, TYPE_ROAD_EDGE_BOUNDARY = 4
            mp_type_mask = mp_token_invalid | ~(mp_token_type[:, :, :5].any(-1))  # [n_sc, n_mp]
            # exclude (3) for veh(0): [n_sc, n_ag, 1] & [n_sc, 1, n_mp]
            attn_mask_veh = ag_type[:, :, [0]] & mp_token_type[:, :, 3].unsqueeze(1)
            # exclude (0,1,2,3) for ped(1): [n_sc, n_ag, 1] & [n_sc, 1, n_mp]
            attn_mask_ped = ag_type[:, :, [1]] & mp_token_type[:, :, :4].any(-1).unsqueeze(1)
            # exclude (0,1,2) for cyc(2): [n_sc, n_ag, 1] & [n_sc, 1, n_mp]
            attn_mask_cyc = ag_type[:, :, [2]] & mp_token_type[:, :, :3].any(-1).unsqueeze(1)
            # [n_sc, n_ag, n_mp]
            logits_invalid = mp_type_mask.unsqueeze(1) | attn_mask_veh | attn_mask_ped | attn_mask_cyc
            logits = logits.masked_fill(logits_invalid, float("-inf"))

            logits_all_inf = logits_invalid.all(-1, keepdim=True)  # [n_sc, n_ag, 1]
            logits = logits.masked_fill(ag_token_invalid.unsqueeze(-1) | logits_all_inf, 0)
            return DestCategorical(logits=logits, valid=ag_token_valid)

        else:  # for goal and cmd
            if self.pairwise_relative:
                rel_pose_ag2mp, rel_dist_ag2mp = get_rel_pose(
                    ag_token_pose, ag_token_invalid, mp_token_pose, mp_token_invalid
                )
            else:
                rel_dist_ag2mp = get_rel_dist(
                    ag_token_pose[..., :2], ag_token_invalid, mp_token_pose[..., :2], mp_token_invalid
                )
                rel_pose_ag2mp = None

            knn_idx_ag2mp, knn_invalid_ag2mp, rpe_ag2mp = get_tgt_knn_idx(
                tgt_invalid=mp_token_invalid,  # [n_sc, n_mp]
                rel_pose=rel_pose_ag2mp,  # [n_sc, n_ag, n_mp, 3] or None
                rel_dist=rel_dist_ag2mp,  # [n_sc, n_ag, n_mp]
                n_tgt_knn=self.n_tgt_knn,
                dist_limit=self.dist_limit,
            )  # knn_idx_ag2mp: [n_sc, n_ag, n_tgt_knn_ag2mp], or None if n_tgt_knn is out of range
            _idx_sc, _idx_ag = torch.arange(n_sc)[:, None, None], torch.arange(n_ag)[None, :, None]
            knn_tgt_ag2mp = mp_token_feature.unsqueeze(1).expand(-1, n_ag, -1, -1)[_idx_sc, _idx_ag, knn_idx_ag2mp]
            if self.pairwise_relative:
                rpe_ag2mp = self.pose_rpe(xy=rpe_ag2mp[..., :2], dir=rpe_ag2mp[..., [2]])

            ag_token_feature, _ = self.tf_ag2mp(
                src=ag_token_feature,  # [n_sc, n_ag, hidden_dim]
                src_padding_mask=ag_token_invalid,  # [n_sc, n_ag]
                tgt=knn_tgt_ag2mp,  # [n_sc, n_ag, n_tgt_knn_ag2mp, hidden_dim]
                tgt_padding_mask=knn_invalid_ag2mp,  # [n_sc, n_ag, n_tgt_knn_ag2mp]
                rpe=rpe_ag2mp,  # [n_sc, n_ag, n_tgt_knn_ag2mp, 3] if pairwise_relative else None
            )  # [n_sc, n_ag, hidden_dim]

            output = self.mlp(ag_token_feature)  # [n_sc, n_ag, nav_dim]

            if self.navi_mode == "goal":
                if self.pairwise_relative:  # transform to global coordinate
                    xy = torch_pos2global(output[:, :, None, :2], ref_pos, ref_rot).squeeze(2)  # [n_sc, n_ag, 2]
                    yaw = torch_rad2global(output[:, :, 2:3], ref_yaw)  # [n_sc, n_ag, 1]
                    output = torch.cat([xy, yaw, output[:, :, 3:4]], dim=-1)
                output = output.masked_fill(ag_token_invalid.unsqueeze(-1), 0)
                return DiagGaussian(mean=output, log_std=self.log_std, valid=ag_token_valid)
            elif self.navi_mode == "cmd":
                output = output.masked_fill(ag_token_invalid.unsqueeze(-1), 0)
                return DestCategorical(logits=output, valid=ag_token_valid)
