# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from typing import Optional, Dict
from torch import Tensor, nn
from omegaconf import DictConfig
from .modules.polyline_encoder import PolylineEncoder
from .modules.input_encoder import InputEncoder
from .modules.transformer_rpe import TransformerBlockRPE
from .modules.multi_agent_gru import MultiAgentGRULoop
from .modules.mlp import MLP
from utils.pose_emb import PoseEmb
from utils.rpe import get_rel_pose, get_rel_dist, get_tgt_knn_idx


class TrafficLightEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tl_state_dim: int,
        pairwise_relative: bool,
        tl_mode: str,
        pose_emb: DictConfig,
        input_encoder: DictConfig,
        pose_rpe: Optional[PoseEmb],
        temp_encoder: DictConfig,
        temp_window_size: int,  # >0 for HPTR, <=0 for TrafficBots RNN
        temp_stack_input: bool,  # for HPTR with temp_window_size>0, stack+MLP instead of temp_encoder
        tf_cfg: DictConfig,
        n_tgt_knn: int,
        k_tgt_knn_tl2tl: float,
        k_tgt_knn_tl2mp: float,
        dist_limit: float,
        k_dist_limit: float,
        n_layer_tf: int,
        tl_lane_detach_mp_feature: int,
    ) -> None:
        super().__init__()
        self.pairwise_relative = pairwise_relative
        self.tl_mode = tl_mode
        self.temp_window_size = temp_window_size
        self.temp_stack_input = temp_stack_input
        self.tl_lane_detach_mp_feature = tl_lane_detach_mp_feature

        if self.tl_mode == "stop":
            if self.pairwise_relative:
                input_pe_dim, self.pose_emb = 0, None
            else:
                pe_dim = hidden_dim if input_encoder.mode == "add" else hidden_dim // 2
                self.pose_emb = PoseEmb(pe_dim=pe_dim, **pose_emb)
                input_pe_dim = self.pose_emb.out_dim
        elif self.tl_mode == "lane":
            input_pe_dim, self.pose_emb = hidden_dim, None

        if self.temp_window_size > 0:
            if self.temp_stack_input:
                attr_dim = tl_state_dim * temp_window_size
            else:
                self.register_buffer("hist_ohe", torch.eye(self.temp_window_size))
                attr_dim = tl_state_dim + temp_window_size
                self.temp_encoder = PolylineEncoder(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **temp_encoder)

            self.n_tgt_knn_tl2tl = int(n_tgt_knn * k_tgt_knn_tl2tl)
            self.n_tgt_knn_tl2mp = int(n_tgt_knn * k_tgt_knn_tl2mp)
            self.dist_limit = dist_limit * k_dist_limit

            self.pose_rpe = pose_rpe
            d_rpe = self.pose_rpe.out_dim if self.pairwise_relative else -1
            self.tf_tl2tlmp = TransformerBlockRPE(n_layer=n_layer_tf, mode="dec_cross_attn", d_rpe=d_rpe, **tf_cfg)
        else:
            attr_dim = tl_state_dim

        self.input_encoder = InputEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim, pe_dim=input_pe_dim, **input_encoder
        )

    def pre_compute(
        self,
        tl_valid: Tensor,  # [n_sc, n_tl]
        tl_attr: Optional[Tensor],  # [n_sc, n_tl] int64 idx if tl_lane, or None if tl_stop
        tl_pose: Tensor,  # [n_sc, n_tl, 3]
        mp_token_invalid: Tensor,  # [n_sc, n_mp]
        mp_token_feature: Tensor,  # [n_sc, n_mp, hidden_dim]
        mp_token_pose: Tensor,  # [n_sc, n_mp, 3]
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Precompute and save static components.
        Returns: Dict tl_tokens
            "tl_token_invalid": [n_sc, n_tl]
            "tl_token_pose": [n_sc, n_tl, 3]
            "tl_token_attr": [n_sc, n_tl, hidden_dim] or None
            "knn_idx_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl] or None
            "knn_invalid_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl] or None
            "rpe_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl, 3] or None
            "knn_tgt_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp] or None
            "knn_invalid_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp] or None
            "rpe_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp, 3] or None
        """
        n_sc, n_tl = tl_valid.shape
        tl_token_invalid = ~tl_valid
        tl_tokens = {
            "tl_token_valid": tl_valid,
            "tl_token_invalid": tl_token_invalid,
            "tl_token_pose": tl_pose,
            "tl_token_attr": None,
            "knn_idx_tl2tl": None,
            "knn_invalid_tl2tl": None,
            "rpe_tl2tl": None,
            "knn_tgt_tl2mp": None,
            "knn_invalid_tl2mp": None,
            "rpe_tl2mp": None,
        }

        if self.tl_mode == "lane":  # Indexing into mp_token_feature, in global (if sc) or local (if pr) coordinate.
            if self.tl_lane_detach_mp_feature:
                mp_token_feature = mp_token_feature.detach()
            tl_tokens["tl_token_attr"] = mp_token_feature[torch.arange(n_sc).unsqueeze(1), tl_attr]

        if self.temp_window_size > 0:  # HPTR tl_token, attending to map
            if self.pairwise_relative:
                rel_pose_tl2tl, rel_dist_tl2tl = get_rel_pose(tl_pose, tl_token_invalid)
                rel_pose_tl2mp, rel_dist_tl2mp = get_rel_pose(
                    tl_pose, tl_token_invalid, mp_token_pose, mp_token_invalid
                )
            else:
                rel_dist_tl2tl = get_rel_dist(tl_pose[..., :2], tl_token_invalid)
                rel_dist_tl2mp = get_rel_dist(
                    tl_pose[..., :2], tl_token_invalid, mp_token_pose[..., :2], mp_token_invalid
                )
                rel_pose_tl2tl, rel_pose_tl2mp = None, None

            tl_tokens["knn_idx_tl2tl"], tl_tokens["knn_invalid_tl2tl"], rpe_tl2tl = get_tgt_knn_idx(
                tgt_invalid=tl_token_invalid,  # [n_sc, n_tl]
                rel_pose=rel_pose_tl2tl,  # [n_sc, n_tl, n_tl, 3] or None
                rel_dist=rel_dist_tl2tl,  # [n_sc, n_tl, n_tl]
                n_tgt_knn=self.n_tgt_knn_tl2tl,
                dist_limit=self.dist_limit,
            )  # knn_idx_tl2tl: [n_sc, n_tl, n_tgt_knn_tl2tl]

            knn_idx_tl2mp, tl_tokens["knn_invalid_tl2mp"], rpe_tl2mp = get_tgt_knn_idx(
                tgt_invalid=mp_token_invalid,  # [n_sc, n_mp]
                rel_pose=rel_pose_tl2mp,  # [n_sc, n_tl, n_mp, 3] or None
                rel_dist=rel_dist_tl2mp,  # [n_sc, n_tl, n_mp]
                n_tgt_knn=self.n_tgt_knn_tl2mp,
                dist_limit=self.dist_limit,
            )  # knn_idx_tl2mp: [n_sc, n_tl, n_tgt_knn_tl2mp]
            tl_tokens["knn_tgt_tl2mp"] = mp_token_feature.unsqueeze(1).expand(-1, n_tl, -1, -1)[
                torch.arange(n_sc)[:, None, None], torch.arange(n_tl)[None, :, None], knn_idx_tl2mp
            ]

            if self.pairwise_relative:
                tl_tokens["rpe_tl2tl"] = self.pose_rpe(xy=rpe_tl2tl[..., :2], dir=rpe_tl2tl[..., [2]])
                tl_tokens["rpe_tl2mp"] = self.pose_rpe(xy=rpe_tl2mp[..., :2], dir=rpe_tl2mp[..., [2]])

        return tl_tokens

    def _get_tl_feature(self, tl_state: Tensor, tl_attr: Optional[Tensor], tl_pose: Tensor) -> Tensor:
        """Feedforward network combine tl_attr and tl_state. Without temporal or token interaction.

        Args:
            tl_state: [n_sc, n_tl, (n_step), tl_state_dim]
            tl_attr: [n_sc, n_tl, hidden_dim], or None
            tl_pose: [n_sc, n_tl, 3]

        Returns:
            tl_feature: [n_sc, n_tl, (n_step), hidden_dim]
        """
        tl_state = tl_state.type_as(tl_pose)

        if self.tl_mode == "stop":
            if self.pairwise_relative:
                tl_feature = self.input_encoder(tl_state, None)
            else:
                tl_pose_emb = self.pose_emb(tl_pose[..., :2], tl_pose[..., 2:3])
                if tl_state.ndim == 4:
                    tl_pose_emb = tl_pose_emb.unsqueeze(2).expand(-1, -1, tl_state.shape[2], -1)
                tl_feature = self.input_encoder(tl_state, tl_pose_emb)
        elif self.tl_mode == "lane":
            if tl_state.ndim == 4:
                tl_attr = tl_attr.unsqueeze(2).expand(-1, -1, tl_state.shape[2], -1)
            tl_feature = self.input_encoder(tl_state, tl_attr)

        return tl_feature

    def forward(
        self,
        tl_state: Tensor,  # [n_sc, n_tl, n_step, tl_state_dim]
        tl_token_invalid: Tensor,  # [n_sc, n_tl]
        tl_token_attr: Optional[Tensor],  # [n_sc, n_tl, hidden_dim] or None
        tl_token_pose: Tensor,  # [n_sc, n_tl, 3]
        knn_idx_tl2tl: Optional[Tensor],  # [n_sc, n_tl, n_tgt_knn_tl2tl] or None
        knn_invalid_tl2tl: Optional[Tensor],  # [n_sc, n_tl, n_tgt_knn_tl2tl] or None
        rpe_tl2tl: Optional[Tensor],  # [n_sc, n_tl, n_tgt_knn_tl2tl, 3] or None
        knn_tgt_tl2mp: Optional[Tensor],  #  [n_sc, n_tl, n_tgt_knn_tl2mp, hidden_dim] or None
        knn_invalid_tl2mp: Optional[Tensor],  # [n_sc, n_tl, n_tgt_knn_tl2mp] or None
        rpe_tl2mp: Optional[Tensor],  # [n_sc, n_tl, n_tgt_knn_tl2mp, 3] or None
        called_by_latent_encoder: bool = False,
        **kwargs,
    ) -> Tensor:
        """if self.self.temp_window_size>0, make tl temporal tokens, temporal aggregation and tl2mptl attention.

        Returns:
            if self.self.temp_window_size>0: # HPTR
                tl_token_feature: [n_sc, n_tl, hidden_dim] # no matter called_by_latent_encoder is True of False
            else: # TrafficBots
                if called_by_latent_encoder:
                    tl_token_feature: [n_sc, n_tl, n_step, hidden_dim]
                else:
                    tl_token_feature: [n_sc, n_tl, hidden_dim]
        """
        if self.temp_window_size > 0:  # HPTR temporal tokens, for latent encoder and rollout
            n_sc, n_tl, n_step, _ = tl_state.shape
            assert n_step <= self.temp_window_size

            # ! [n_sc, n_tl, n_step, hidden_dim] -> [n_sc, n_tl, hidden_dim]
            if self.temp_stack_input:
                if n_step < self.temp_window_size:  # zero padding from left
                    tl_state = torch.nn.functional.pad(
                        tl_state, (0, 0, self.temp_window_size - n_step, 0), "constant", 0
                    )  # [n_sc, n_tl, self.temp_window_size, tl_state_dim]
                tl_state = tl_state.flatten(2, 3)  # [n_sc, n_tl, self.temp_window_size * tl_state_dim]
                tl_token_feature = self._get_tl_feature(tl_state, tl_token_attr, tl_token_pose)
            else:
                tl_state = torch.cat(
                    [tl_state, self.hist_ohe[None, None, -n_step:, :].expand(n_sc, n_tl, -1, -1)], dim=-1
                )
                tl_feature = self._get_tl_feature(tl_state, tl_token_attr, tl_token_pose)
                # aggregate temporal axis for each traffic light
                tl_token_feature = self.temp_encoder(tl_feature, tl_token_invalid.unsqueeze(-1).expand(-1, -1, n_step))

            # ! decoder cross attention to map
            tl_token_feature, _ = self.tf_tl2tlmp(
                src=tl_token_feature,  # [n_sc, n_tl, hidden_dim]
                src_padding_mask=tl_token_invalid,  # [n_sc, n_tl]
                tgt=knn_tgt_tl2mp,  # [n_sc, n_tl, n_tgt_knn_tl2mp, hidden_dim]
                tgt_padding_mask=knn_invalid_tl2mp,  # [n_sc, n_tl, n_tgt_knn_tl2mp]
                rpe=rpe_tl2mp,  # [n_sc, n_tl, n_tgt_knn_tl2mp, 3] if pairwise_relative else None
                decoder_tgt=knn_idx_tl2tl,  # [n_sc, n_tl, n_tgt_knn_tl2tl]
                decoder_tgt_padding_mask=knn_invalid_tl2tl,  # [n_sc, n_tl, n_tgt_knn_tl2tl]
                decoder_rpe=rpe_tl2tl,  # [n_sc, n_tl, n_tgt_knn_tl2tl, 3] if pairwise_relative else None
            )
        else:
            if not called_by_latent_encoder:  # TrafficBots RNN, consider only the latest step
                tl_state = tl_state[:, :, -1]
            tl_token_feature = self._get_tl_feature(tl_state, tl_token_attr, tl_token_pose)

        return tl_token_feature


class TrafficLightStatePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tl_state_dim: int,
        n_layer: int,
        rnn_dropout_p: float,
        temp_window_size: bool,
        detach_tl_feature: bool,
    ) -> None:
        super().__init__()
        self.temp_window_size = temp_window_size
        self.detach_tl_feature = detach_tl_feature
        if self.temp_window_size <= 0:
            self.rnn = MultiAgentGRULoop(hidden_dim, n_layer, rnn_dropout_p)

        self.mlp = MLP([hidden_dim] * n_layer + [tl_state_dim], end_layer_activation=False)

    def init(self) -> None:
        self.rnn_hidden = None

    def forward(self, tl_token_feature: Tensor, tl_token_invalid: Tensor) -> Tensor:
        """
        Args:
            tl_token_feature: [n_sc, n_tl, hidden_dim]
            tl_token_invalid: [n_sc, n_tl]

        Returns:
            tl_state_pred_logits: [n_sc, n_tl, tl_state_dim], logits
        """
        if self.detach_tl_feature:
            tl_token_feature = tl_token_feature.detach()
        if self.temp_window_size <= 0:
            tl_token_feature, self.rnn_hidden = self.rnn(
                tl_token_feature, torch.zeros_like(tl_token_feature[:, :, 0], dtype=bool), self.rnn_hidden
            )
        tl_state_pred_logits = self.mlp(tl_token_feature, tl_token_invalid)
        return torch.clamp(tl_state_pred_logits, min=-3, max=3)

