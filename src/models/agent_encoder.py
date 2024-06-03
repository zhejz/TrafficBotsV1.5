# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from typing import Optional, Dict, Tuple
from torch import Tensor, nn
from omegaconf import DictConfig
from .modules.polyline_encoder import PolylineEncoder
from .modules.input_encoder import InputEncoder
from .modules.multi_agent_gru import MultiAgentGRULoop
from .modules.transformer_rpe import TransformerBlockRPE
from utils.pooling import seq_pooling
from utils.pose_emb import PoseEmb
from utils.rpe import get_rel_pose, get_rel_dist, get_tgt_knn_idx
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_rad2local


class AgentEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ag_attr_dim: int,
        ag_motion_dim: int,
        pairwise_relative: bool,
        pose_emb: DictConfig,
        input_encoder: DictConfig,
        temp_encoder: DictConfig,
        pose_rpe: PoseEmb,
        tf_cfg: DictConfig,
        n_tgt_knn: int,
        k_tgt_knn_ag2ag: float,
        k_tgt_knn_ag2mp: float,
        k_tgt_knn_ag2tl: float,
        dist_limit: float,
        k_dist_limit: float,
        n_layer_tf: int,
        temp_window_size: int,  # >0 for HPTR, <=0 for TrafficBots RNN
        rnn_latent_temp_pool_mode: str,
    ) -> None:
        super().__init__()
        self.pairwise_relative = pairwise_relative
        self.temp_window_size = temp_window_size
        self.n_tgt_knn_ag2ag = int(n_tgt_knn * k_tgt_knn_ag2ag)
        self.n_tgt_knn_ag2mp = int(n_tgt_knn * k_tgt_knn_ag2mp)
        self.n_tgt_knn_ag2tl = int(n_tgt_knn * k_tgt_knn_ag2tl)
        self.dist_limit = dist_limit * k_dist_limit

        if self.temp_window_size <= 0 and self.pairwise_relative:  # pairwise-relative RNN
            input_pe_dim = 0  # no global/local pose, just the difference
            self.pose_emb = None
        else:
            pe_dim = hidden_dim if input_encoder.mode == "add" else hidden_dim // 2
            self.pose_emb = PoseEmb(pe_dim=pe_dim, **pose_emb)
            input_pe_dim = self.pose_emb.out_dim

        self.pose_rpe = pose_rpe
        d_rpe = self.pose_rpe.out_dim if self.pairwise_relative else -1
        attr_dim = ag_attr_dim + ag_motion_dim
        if self.temp_window_size > 0:
            self.register_buffer("hist_ohe", torch.eye(self.temp_window_size))
            attr_dim += self.temp_window_size
            self.temp_encoder = PolylineEncoder(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **temp_encoder)
            self.tf_ag2agmptl = TransformerBlockRPE(n_layer=n_layer_tf, mode="dec_cross_attn", d_rpe=d_rpe, **tf_cfg)
        else:
            self.tf_ag2mp = TransformerBlockRPE(n_layer=n_layer_tf, mode="enc_cross_attn", d_rpe=d_rpe, **tf_cfg)
            self.tf_ag2tl = TransformerBlockRPE(n_layer=n_layer_tf, mode="enc_cross_attn", d_rpe=d_rpe, **tf_cfg)
            self.tf_ag2ag = TransformerBlockRPE(n_layer=n_layer_tf, mode="enc_self_attn", d_rpe=d_rpe, **tf_cfg)
            self.temp_encoder = MultiAgentGRULoop(hidden_dim, temp_encoder["n_layer"], temp_encoder["mlp_dropout_p"])
            self.rnn_latent_temp_pool_mode = rnn_latent_temp_pool_mode

        self.input_encoder = InputEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim, pe_dim=input_pe_dim, **input_encoder
        )

    def forward(self, called_by_latent_encoder=False, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """if self.temp_window_size>0, make ag temporal tokens, temporal aggregation and ag2agmptl attention.

        Args:
            ag_valid: Tensor: [n_sc, n_ag, n_step]
            ag_attr: Tensor: [n_sc, n_ag, ag_attr_dim]
            ag_motion: Tensor: [n_sc, n_ag, n_step, ag_motion_dim]
            ag_pose: Tensor: [n_sc, n_ag, n_step, 3]
            mp_token_invalid: Tensor: [n_sc, n_mp]
            mp_token_feature: Tensor: [n_sc, n_mp, hidden_dim]
            mp_token_pose: Tensor: [n_sc, n_mp, 3]
            mp_token_type: [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11
            tl_token_invalid: Tensor: [n_sc, n_tl]
            tl_token_pose: Tensor: [n_sc, n_tl, 3]
            rnn_hidden: Optional[Tensor]: None if self.temp_window_size>0
            called_by_latent_encoder: bool = False

            tl_token_feature: Tensor: [n_sc, n_tl, (n_step), hidden_dim]
            if self.temp_window_size>0: # HPTR
                tl_token_feature: [n_sc, n_tl, hidden_dim] # no matter called_by_latent_encoder is True of False
            else: # TrafficBots
                if called_by_latent_encoder:
                    tl_token_feature: [n_sc, n_tl, n_step, hidden_dim]
                else:
                    tl_token_feature: [n_sc, n_tl, hidden_dim]
                    
        Returns:
            ag_token_feature: [n_sc, n_ag, hidden_dim]
            rnn_hidden: [n_layer, n_sc*n_ag, hidden_dim] or None
        """
        if self.temp_window_size > 0:  # HPTR temporal tokens, for latent encoder and rollout
            ag_token_feature = self._forward_hptr(**kwargs)
            rnn_hidden = None
        else:
            if called_by_latent_encoder:
                ag_token_feature = self._forward_trafficbots_latent_encoder(**kwargs)
                rnn_hidden = None
            else:  # TrafficBots RNN, consider only the latest step
                ag_token_feature, rnn_hidden = self._forward_trafficbots_rollout(**kwargs)
        return ag_token_feature, rnn_hidden

    def _forward_hptr(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag, n_step]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_motion: Tensor,  # [n_sc, n_ag, n_step, ag_motion_dim]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_feature: Tensor,  # [n_sc, n_tl, hidden_dim]
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
        **kwargs,
    ) -> Tensor:  # ag_token_feature: [n_sc, n_ag, hidden_dim]
        # Make ag temporal tokens, temporal aggregation and ag2agmptl attention.
        # ! pre-compute
        n_sc, n_ag, n_step = ag_valid.shape
        ag_invalid, ag_token_invalid = ~ag_valid, ~(ag_valid.any(-1))
        ag_token_pose = seq_pooling(ag_pose, ag_invalid, "last_valid", ag_valid)
        knn_ag2mp, knn_ag2tl, knn_ag2ag = self._get_knn_for_ag(
            ag_token_invalid=ag_token_invalid,
            ag_token_pose=ag_token_pose,
            mp_token_invalid=mp_token_invalid,
            mp_token_feature=mp_token_feature,
            mp_token_pose=mp_token_pose,
            tl_token_invalid=tl_token_invalid,
            tl_token_feature=tl_token_feature,
            tl_token_pose=tl_token_pose,
        )

        # ! input, combine ag_attr, ag_motion and ag_pose
        ag_xy, ag_yaw = ag_pose[..., :2], ag_pose[..., 2:3]  # [n_sc, n_ag, n_step, 2/1]
        if self.pairwise_relative:
            ag_xy = torch_pos2local(ag_xy, ag_token_pose[:, :, None, :2], torch_rad2rot(ag_token_pose[..., -1]))
            ag_yaw = torch_rad2local(ag_yaw.squeeze(-1), ag_token_pose[..., -1], cast=False).unsqueeze(-1)

        ag_attr = torch.cat(
            [
                ag_attr.unsqueeze(2).expand(-1, -1, n_step, -1),
                ag_motion,
                self.hist_ohe[None, None, -n_step:, :].expand(n_sc, n_ag, -1, -1),
            ],
            dim=-1,
        )

        ag_feature = self.input_encoder(ag_attr, self.pose_emb(ag_xy, ag_yaw))  # [n_sc, n_ag, n_step, hidden_dim]

        # ! aggregate temporal axis for each agent
        ag_token_feature = self.temp_encoder(ag_feature, ag_invalid)  # [n_sc, n_ag, hidden_dim]

        # ! decoder cross attention to map and traffic lights
        knn_tgt_ag2mptl = torch.cat([knn_ag2mp["tgt"], knn_ag2tl["tgt"]], dim=2)
        knn_invalid_ag2mptl = torch.cat([knn_ag2mp["invalid"], knn_ag2tl["invalid"]], dim=2)
        rpe_ag2mptl = torch.cat([knn_ag2mp["rpe"], knn_ag2tl["rpe"]], dim=2) if self.pairwise_relative else None
        ag_token_feature, _ = self.tf_ag2agmptl(
            src=ag_token_feature,  # [n_sc, n_ag, hidden_dim]
            src_padding_mask=ag_token_invalid,  # [n_sc, n_ag]
            tgt=knn_tgt_ag2mptl,  # [n_sc, n_ag, n_tgt_knn_ag2mp+n_tgt_knn_ag2tl, hidden_dim]
            tgt_padding_mask=knn_invalid_ag2mptl,  # [n_sc, n_ag, n_tgt_knn_ag2mp+n_tgt_knn_ag2tl]
            rpe=rpe_ag2mptl,  # [n_sc, n_ag, n_tgt_knn_ag2mp+n_tgt_knn_ag2tl, 3] if pairwise_relative else None
            decoder_tgt=knn_ag2ag["idx"],  # [n_sc, n_ag, n_tgt_knn_ag2ag]
            decoder_tgt_padding_mask=knn_ag2ag["invalid"],  # [n_sc, n_ag, n_tgt_knn_ag2ag]
            decoder_rpe=knn_ag2ag["rpe"],  # [n_sc, n_ag, n_tgt_knn_ag2ag, 3] if pairwise_relative else None
        )
        return ag_token_feature

    def _forward_trafficbots_rollout(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag, n_step]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_motion: Tensor,  # [n_sc, n_ag, n_step, ag_motion_dim]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_feature: Tensor,  # [n_sc, n_tl, hidden_dim]
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
        rnn_hidden: Optional[Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            ag_token_feature: [n_sc, n_ag, hidden_dim]
            rnn_hidden: [n_layer, n_sc*n_ag, hidden_dim]
        """
        # ! pre-compute
        ag_token_pose, ag_token_invalid = ag_pose[:, :, -1], ~ag_valid[:, :, -1]
        knn_ag2mp, knn_ag2tl, knn_ag2ag = self._get_knn_for_ag(
            ag_token_invalid=ag_token_invalid,
            ag_token_pose=ag_token_pose,
            mp_token_invalid=mp_token_invalid,
            mp_token_feature=mp_token_feature,
            mp_token_pose=mp_token_pose,
            tl_token_invalid=tl_token_invalid,
            tl_token_feature=tl_token_feature,
            tl_token_pose=tl_token_pose,
        )

        # ! input, combine ag_attr, ag_motion and ag_pose
        if self.pairwise_relative:
            ag_pose_emb = None
        else:
            ag_pose_emb = self.pose_emb(ag_token_pose[..., :2], ag_token_pose[..., 2:3])  # [n_sc, n_ag, 2/1]

        ag_token_feature = self.input_encoder(torch.cat([ag_attr, ag_motion[:, :, -1]], dim=-1), ag_pose_emb)

        # ! encoder cross attention to map
        ag_token_feature, _ = self.tf_ag2mp(
            src=ag_token_feature,  # [n_sc, n_ag, hidden_dim]
            src_padding_mask=ag_token_invalid,  # [n_sc, n_ag]
            tgt=knn_ag2mp["tgt"],  # [n_sc, n_ag, n_tgt_knn_ag2mp, hidden_dim]
            tgt_padding_mask=knn_ag2mp["invalid"],  # [n_sc, n_ag, n_tgt_knn_ag2mp]
            rpe=knn_ag2mp["rpe"],  # [n_sc, n_ag, n_tgt_knn_ag2mp, 3] if pairwise_relative else None
        )
        # ! encoder cross attention to traffic light
        ag_token_feature, _ = self.tf_ag2tl(
            src=ag_token_feature,  # [n_sc, n_ag, hidden_dim]
            src_padding_mask=ag_token_invalid,  # [n_sc, n_ag]
            tgt=knn_ag2tl["tgt"],  # [n_sc, n_ag, n_tgt_knn_ag2tl, hidden_dim]
            tgt_padding_mask=knn_ag2tl["invalid"],  # [n_sc, n_ag, n_tgt_knn_ag2tl]
            rpe=knn_ag2tl["rpe"],  # [n_sc, n_ag, n_tgt_knn_ag2tl, 3] if pairwise_relative else None
        )
        # ! encoder self attention to agents
        ag_token_feature, _ = self.tf_ag2ag(
            src=ag_token_feature,  # [n_sc, n_ag, hidden_dim]
            src_padding_mask=ag_token_invalid,  # [n_sc, n_ag]
            tgt=knn_ag2ag["idx"],  # [n_sc, n_ag, n_tgt_knn_ag2ag]
            tgt_padding_mask=knn_ag2ag["invalid"],  # [n_sc, n_ag, n_tgt_knn_ag2ag]
            rpe=knn_ag2ag["rpe"],  # [n_sc, n_ag, n_tgt_knn_ag2ag, 3] if pairwise_relative else None
        )

        # ! temporal encoder
        ag_token_feature, rnn_hidden = self.temp_encoder(ag_token_feature, ag_token_invalid, rnn_hidden)

        return ag_token_feature, rnn_hidden

    def _forward_trafficbots_latent_encoder(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag, n_step]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_motion: Tensor,  # [n_sc, n_ag, n_step, ag_motion_dim]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_feature: Tensor,  # [n_sc, n_tl, n_step, hidden_dim]
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
        **kwargs,
    ) -> Tensor:  # ag_token_feature: [n_sc, n_ag, hidden_dim]
        n_sc, n_ag, n_step = ag_valid.shape
        # ! pre-compute
        ag_invalid = ~ag_valid
        ag_invalid_flat_sc_step = ag_invalid.transpose(1, 2).flatten(0, 1)  # [n_sc*n_step n_ag]
        knn_ag2mp, knn_ag2tl, knn_ag2ag = self._get_knn_for_ag_trafficbots_latent(
            ag_invalid=ag_invalid,
            ag_pose=ag_pose,
            mp_token_invalid=mp_token_invalid,
            mp_token_feature=mp_token_feature,
            mp_token_pose=mp_token_pose,
            tl_token_invalid=tl_token_invalid,
            tl_token_feature=tl_token_feature,
            tl_token_pose=tl_token_pose,
        )

        # ! input, combine ag_attr, ag_motion and ag_pose
        if self.pairwise_relative:
            ag_pose_emb = None
        else:
            ag_pose_emb = self.pose_emb(ag_pose[..., :2], ag_pose[..., 2:3])  # [n_sc, n_ag, n_step, 2/1]

        ag_token_feature = self.input_encoder(
            torch.cat([ag_attr.unsqueeze(2).expand(-1, -1, n_step, -1), ag_motion], dim=-1), ag_pose_emb
        )  # [n_sc, n_ag, n_step, hidden_dim]

        # ! encoder cross attention to map
        ag_token_feature, _ = self.tf_ag2mp(
            src=ag_token_feature.flatten(1, 2),  # [n_sc, n_ag*n_step, hidden_dim]
            src_padding_mask=ag_invalid.flatten(1, 2),  # [n_sc, n_ag*n_step]
            tgt=knn_ag2mp["tgt"],  # [n_sc, n_ag*n_step, n_tgt_knn_ag2mp, hidden_dim]
            tgt_padding_mask=knn_ag2mp["invalid"],  # [n_sc, n_ag*n_step, n_tgt_knn_ag2mp]
            rpe=knn_ag2mp["rpe"],  # [n_sc, n_ag*n_step, n_tgt_knn_ag2mp, 3] if pairwise_relative else None
        )
        ag_token_feature = ag_token_feature.view(n_sc, n_ag, n_step, -1)
        # ! encoder cross attention to traffic light
        ag_token_feature, _ = self.tf_ag2tl(
            src=ag_token_feature.transpose(1, 2).flatten(0, 1),  # [n_sc*n_step, n_ag, hidden_dim]
            src_padding_mask=ag_invalid_flat_sc_step,  # [n_sc*n_step, n_ag]
            tgt=knn_ag2tl["tgt"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2tl, hidden_dim]
            tgt_padding_mask=knn_ag2tl["invalid"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2tl]
            rpe=knn_ag2tl["rpe"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2tl, 3] if pairwise_relative else None
        )
        # ! encoder self attention to agents
        ag_token_feature, _ = self.tf_ag2ag(
            src=ag_token_feature,  # [n_sc*n_step, n_ag, hidden_dim]
            src_padding_mask=ag_invalid_flat_sc_step,  # [n_sc*n_step, n_ag]
            tgt=knn_ag2ag["idx"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2ag]
            tgt_padding_mask=knn_ag2ag["invalid"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2ag]
            rpe=knn_ag2ag["rpe"],  # [n_sc*n_step, n_ag, n_tgt_knn_ag2ag, 3] if pairwise_relative else None
        )
        ag_token_feature = ag_token_feature.view(n_sc, n_step, n_ag, -1).transpose(1, 2)
        # ! temporal encoder
        ag_token_feature, _ = self.temp_encoder(ag_token_feature, ag_invalid)
        ag_token_feature = seq_pooling(ag_token_feature, ag_invalid, self.rnn_latent_temp_pool_mode, ag_valid)
        return ag_token_feature

    def _get_knn_for_ag(
        self,
        ag_token_invalid: Tensor,  # [n_sc, n_ag]
        ag_token_pose: Tensor,  # [n_sc, n_ag, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_feature: Tensor,  # [n_sc, n_tl, hidden_dim]
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        knn_ag2mp = {"tgt": None, "invalid": None, "rpe": None}
        knn_ag2tl = {"tgt": None, "invalid": None, "rpe": None}
        knn_ag2ag = {"idx": None, "invalid": None, "rpe": None}
        n_sc, n_ag = ag_token_invalid.shape
        _idx_sc, _idx_ag = torch.arange(n_sc)[:, None, None], torch.arange(n_ag)[None, :, None]

        if self.pairwise_relative:
            rel_pose_ag2ag, rel_dist_ag2ag = get_rel_pose(ag_token_pose, ag_token_invalid)
            rel_pose_ag2mp, rel_dist_ag2mp = get_rel_pose(
                ag_token_pose, ag_token_invalid, mp_token_pose, mp_token_invalid
            )
            rel_pose_ag2tl, rel_dist_ag2tl = get_rel_pose(
                ag_token_pose, ag_token_invalid, tl_token_pose, tl_token_invalid
            )
        else:
            rel_dist_ag2ag = get_rel_dist(ag_token_pose[..., :2], ag_token_invalid)
            rel_dist_ag2mp = get_rel_dist(
                ag_token_pose[..., :2], ag_token_invalid, mp_token_pose[..., :2], mp_token_invalid
            )
            rel_dist_ag2tl = get_rel_dist(
                ag_token_pose[..., :2], ag_token_invalid, tl_token_pose[..., :2], tl_token_invalid
            )
            rel_pose_ag2ag, rel_pose_ag2mp, rel_pose_ag2tl = None, None, None
        # ! ag2ag
        knn_ag2ag["idx"], knn_ag2ag["invalid"], rpe_ag2ag = get_tgt_knn_idx(
            tgt_invalid=ag_token_invalid,  # [n_sc, n_ag]
            rel_pose=rel_pose_ag2ag,  # [n_sc, n_ag, n_ag, 3] or None
            rel_dist=rel_dist_ag2ag,  # [n_sc, n_ag, n_ag]
            n_tgt_knn=self.n_tgt_knn_ag2ag,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2ag: [n_sc, n_ag, n_tgt_knn_ag2ag]
        # ! ag2mp
        knn_idx_ag2mp, knn_ag2mp["invalid"], rpe_ag2mp = get_tgt_knn_idx(
            tgt_invalid=mp_token_invalid,  # [n_sc, n_mp]
            rel_pose=rel_pose_ag2mp,  # [n_sc, n_ag, n_mp, 3] or None
            rel_dist=rel_dist_ag2mp,  # [n_sc, n_ag, n_mp]
            n_tgt_knn=self.n_tgt_knn_ag2mp,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2mp: [n_sc, n_ag, n_tgt_knn_ag2mp]
        knn_ag2mp["tgt"] = mp_token_feature.unsqueeze(1).expand(-1, n_ag, -1, -1)[_idx_sc, _idx_ag, knn_idx_ag2mp]
        # ! ag2tl
        knn_idx_ag2tl, knn_ag2tl["invalid"], rpe_ag2tl = get_tgt_knn_idx(
            tgt_invalid=tl_token_invalid,  # [n_sc, n_tl]
            rel_pose=rel_pose_ag2tl,  # [n_sc, n_ag, n_tl, 3] or None
            rel_dist=rel_dist_ag2tl,  # [n_sc, n_ag, n_tl]
            n_tgt_knn=self.n_tgt_knn_ag2tl,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2tl: [n_sc, n_ag, n_tgt_knn_ag2tl]
        knn_ag2tl["tgt"] = tl_token_feature.unsqueeze(1).expand(-1, n_ag, -1, -1)[_idx_sc, _idx_ag, knn_idx_ag2tl]

        if self.pairwise_relative:
            knn_ag2ag["rpe"] = self.pose_rpe(xy=rpe_ag2ag[..., :2], dir=rpe_ag2ag[..., [2]])
            knn_ag2mp["rpe"] = self.pose_rpe(xy=rpe_ag2mp[..., :2], dir=rpe_ag2mp[..., [2]])
            knn_ag2tl["rpe"] = self.pose_rpe(xy=rpe_ag2tl[..., :2], dir=rpe_ag2tl[..., [2]])

        return knn_ag2mp, knn_ag2tl, knn_ag2ag

    def _get_knn_for_ag_trafficbots_latent(
        self,
        ag_invalid: Tensor,  # [n_sc, n_ag, n_step]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_feature: Tensor,  # [n_sc, n_tl, n_step, hidden_dim]
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        knn_ag2mp = {"tgt": None, "invalid": None, "rpe": None}  # [n_sc, n_ag*n_step, n_tgt]
        knn_ag2tl = {"tgt": None, "invalid": None, "rpe": None}  # [n_sc*n_step, n_ag, n_tgt]
        knn_ag2ag = {"idx": None, "invalid": None, "rpe": None}  # [n_sc*n_step, n_ag, n_tgt]
        n_sc, n_ag, n_step = ag_invalid.shape
        _idx_sc, _idx_ag = torch.arange(n_sc)[:, None, None], torch.arange(n_ag)[None, :, None]
        _idx_sc_step = torch.arange(n_sc * n_step)[:, None, None]
        _idx_ag_step = torch.arange(n_ag * n_step)[None, :, None]

        ag_pose_flat_sc_step = ag_pose.transpose(1, 2).flatten(0, 1)  # [n_sc*n_step, n_ag, hidden_dim]
        ag_invalid_flat_sc_step = ag_invalid.transpose(1, 2).flatten(0, 1)  # [n_sc*n_step, n_ag]
        tl_token_feature = tl_token_feature.transpose(1, 2).flatten(0, 1)  # [n_sc*n_step, n_tl, hidden_dim]
        tl_token_pose = tl_token_pose.unsqueeze(1).expand(-1, n_step, -1, -1).flatten(0, 1)  # [n_sc*n_step, n_tl, 3]
        tl_token_invalid = tl_token_invalid.unsqueeze(1).expand(-1, n_step, -1).flatten(0, 1)  # [n_sc*n_step, n_tl]

        if self.pairwise_relative:
            rel_pose_ag2mp, rel_dist_ag2mp = get_rel_pose(
                ag_pose.flatten(1, 2), ag_invalid.flatten(1, 2), mp_token_pose, mp_token_invalid
            )
            rel_pose_ag2tl, rel_dist_ag2tl = get_rel_pose(
                ag_pose_flat_sc_step, ag_invalid_flat_sc_step, tl_token_pose, tl_token_invalid
            )
            rel_pose_ag2ag, rel_dist_ag2ag = get_rel_pose(ag_pose_flat_sc_step, ag_invalid_flat_sc_step)
        else:
            rel_dist_ag2mp = get_rel_dist(
                ag_pose[..., :2].flatten(1, 2), ag_invalid.flatten(1, 2), mp_token_pose[..., :2], mp_token_invalid
            )
            rel_dist_ag2tl = get_rel_dist(
                ag_pose_flat_sc_step[..., :2], ag_invalid_flat_sc_step, tl_token_pose[..., :2], tl_token_invalid
            )
            rel_dist_ag2ag = get_rel_dist(ag_pose_flat_sc_step[..., :2], ag_invalid_flat_sc_step)
            rel_pose_ag2mp, rel_pose_ag2tl, rel_pose_ag2ag = None, None, None

        # ! ag2mp
        knn_idx_ag2mp, knn_ag2mp["invalid"], rpe_ag2mp = get_tgt_knn_idx(
            tgt_invalid=mp_token_invalid,  # [n_sc, n_mp]
            rel_pose=rel_pose_ag2mp,  # [n_sc, n_ag*n_step, n_mp, 3] or None
            rel_dist=rel_dist_ag2mp,  # [n_sc, n_ag*n_step, n_mp]
            n_tgt_knn=self.n_tgt_knn_ag2mp,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2mp: [n_sc, n_ag*n_step, n_tgt_knn_ag2mp]
        knn_ag2mp["tgt"] = mp_token_feature.unsqueeze(1).expand(-1, n_ag * n_step, -1, -1)[
            _idx_sc, _idx_ag_step, knn_idx_ag2mp
        ]
        # ! ag2tl
        knn_idx_ag2tl, knn_ag2tl["invalid"], rpe_ag2tl = get_tgt_knn_idx(
            tgt_invalid=tl_token_invalid,  # [n_sc*n_step, n_tl]
            rel_pose=rel_pose_ag2tl,  # [n_sc*n_step, n_ag, n_tl, 3] or None
            rel_dist=rel_dist_ag2tl,  # [n_sc*n_step, n_ag, n_tl]
            n_tgt_knn=self.n_tgt_knn_ag2tl,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2tl: [n_sc*n_step, n_ag, n_tgt_knn_ag2tl]
        knn_ag2tl["tgt"] = tl_token_feature.unsqueeze(1).expand(-1, n_ag, -1, -1)[_idx_sc_step, _idx_ag, knn_idx_ag2tl]
        # ! ag2ag
        knn_ag2ag["idx"], knn_ag2ag["invalid"], rpe_ag2ag = get_tgt_knn_idx(
            tgt_invalid=ag_invalid_flat_sc_step,  # [n_sc*n_step, n_ag]
            rel_pose=rel_pose_ag2ag,  # [n_sc*n_step, n_ag, n_ag, 3] or None
            rel_dist=rel_dist_ag2ag,  # [n_sc*n_step, n_ag, n_ag]
            n_tgt_knn=self.n_tgt_knn_ag2ag,
            dist_limit=self.dist_limit,
        )  # knn_idx_ag2ag: [n_sc*n_step, n_ag, n_tgt_knn_ag2ag]

        if self.pairwise_relative:
            knn_ag2mp["rpe"] = self.pose_rpe(xy=rpe_ag2mp[..., :2], dir=rpe_ag2mp[..., [2]])
            knn_ag2tl["rpe"] = self.pose_rpe(xy=rpe_ag2tl[..., :2], dir=rpe_ag2tl[..., [2]])
            knn_ag2ag["rpe"] = self.pose_rpe(xy=rpe_ag2ag[..., :2], dir=rpe_ag2ag[..., [2]])

        return knn_ag2mp, knn_ag2tl, knn_ag2ag
