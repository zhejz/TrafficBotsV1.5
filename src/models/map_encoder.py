# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from typing import Dict
from torch import Tensor, nn
from omegaconf import DictConfig
from .modules.polyline_encoder import PolylineEncoder
from .modules.input_encoder import InputEncoder
from .modules.transformer_rpe import TransformerBlockRPE
from utils.pose_emb import PoseEmb
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_rad2local
from utils.rpe import get_rel_pose, get_rel_dist, get_tgt_knn_idx


class MapEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pairwise_relative: bool,
        pose_emb: DictConfig,
        n_mp_pl_node: int,
        input_encoder: DictConfig,
        pl_encoder: DictConfig,
        pose_rpe: nn.Module,
        tf_cfg: DictConfig,
        n_layer_tf: int,
        n_tgt_knn: int,
        dist_limit: float,
    ) -> None:
        super().__init__()
        self.pairwise_relative = pairwise_relative
        self.pose_rpe = pose_rpe
        self.n_tgt_knn = n_tgt_knn
        self.dist_limit = dist_limit

        self.register_buffer("pl_node_ohe", torch.eye(n_mp_pl_node)[None, None, :, :])

        pe_dim = hidden_dim if input_encoder.mode == "add" else hidden_dim // 2
        self.pose_emb = PoseEmb(pe_dim=pe_dim, **pose_emb)

        self.input_encoder = InputEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim + n_mp_pl_node, pe_dim=self.pose_emb.out_dim, **input_encoder
        )

        self.pl_encoder = PolylineEncoder(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **pl_encoder)

        d_rpe = self.pose_rpe.out_dim if self.pairwise_relative else -1
        self.tf_mp2mp = TransformerBlockRPE(n_layer=n_layer_tf, mode="enc_self_attn", d_rpe=d_rpe, **tf_cfg)

    def forward(self, mp_valid: Tensor, mp_attr: Tensor, mp_pose: Tensor, mp_type: Tensor) -> Dict[str, Tensor]:
        """Aggregate polyline-level feature. n_mp_pl_node nodes per polyline. c.f. VectorNet, SceneTransformer
        
        Args: in scene-centric coordinate
            mp_valid: [n_sc, n_mp, n_mp_pl_node]
            mp_attr: [n_sc, n_mp, mp_attr_dim]
            mp_pose: [n_sc, n_mp, n_mp_pl_node, 3]
            mp_type: [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11

        Returns: Dict mp_tokens
            "mp_token_invalid": [n_sc, n_mp]
            "mp_token_feature": [n_sc, n_mp, hidden_dim]
            "mp_token_pose": [n_sc, n_mp, 3]
            "mp_token_type": [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11
        """
        mp_token_pose, mp_token_valid = mp_pose[:, :, 0], mp_valid[:, :, 0]
        mp_invalid, mp_token_invalid = ~mp_valid, ~mp_token_valid
        n_sc, n_mp, n_mp_pl_node = mp_valid.shape

        mp_xy, mp_yaw = mp_pose[..., :2], mp_pose[..., 2:3]  # [n_sc, n_mp, n_mp_pl_node, 2/1]
        if self.pairwise_relative:
            mp_xy = torch_pos2local(mp_xy, mp_token_pose[:, :, None, :2], torch_rad2rot(mp_token_pose[..., -1]))
            mp_yaw = torch_rad2local(mp_yaw.squeeze(-1), mp_token_pose[..., -1], cast=False).unsqueeze(-1)
        mp_pose_emb = self.pose_emb(mp_xy, mp_yaw)  # [n_sc, n_mp, n_mp_pl_node, pe_dim]

        mp_attr = torch.cat(
            [mp_attr.unsqueeze(2).expand(-1, -1, n_mp_pl_node, -1), self.pl_node_ohe.expand(n_sc, n_mp, -1, -1)], dim=-1
        )  # [n_sc, n_mp, n_mp_pl_node, attr_dim+n_mp_pl_node]

        mp_pl_feature = self.input_encoder(mp_attr, mp_pose_emb)  # [n_sc, n_mp, n_mp_pl_node, hidden_dim]
        mp_token_feature = self.pl_encoder(mp_pl_feature, mp_invalid)  # [n_sc, n_mp, hidden_dim]

        if self.pairwise_relative:
            rel_pose, rel_dist = get_rel_pose(mp_token_pose, mp_token_invalid)
        else:
            rel_dist = get_rel_dist(mp_token_pose[..., :2], mp_token_invalid)
            rel_pose = None

        knn_idx_mp2mp, knn_invalid_mp2mp, rpe_mp2mp = get_tgt_knn_idx(
            tgt_invalid=mp_token_invalid,  # [n_sc, n_mp]
            rel_pose=rel_pose,  # [n_sc, n_mp, n_mp, 3] or None
            rel_dist=rel_dist,  # [n_sc, n_mp, n_mp]
            n_tgt_knn=self.n_tgt_knn,
            dist_limit=self.dist_limit,
        )

        if self.pairwise_relative:
            rpe_mp2mp = self.pose_rpe(xy=rpe_mp2mp[..., :2], dir=rpe_mp2mp[..., [2]])

        mp_token_feature, _ = self.tf_mp2mp(
            src=mp_token_feature,  # [n_sc, n_mp, hidden_dim]
            src_padding_mask=mp_token_invalid,  # [n_sc, n_mp]
            tgt=knn_idx_mp2mp,  # [n_sc, n_mp, n_tgt_knn]
            tgt_padding_mask=knn_invalid_mp2mp,  # [n_sc, n_mp, n_tgt_knn]
            rpe=rpe_mp2mp,  # [n_sc, n_mp, n_tgt_knn, d_rpe] if pairwise_relative else None
        )

        mp_tokens = {
            "mp_token_invalid": mp_token_invalid,  # [n_sc, n_mp]
            "mp_token_feature": mp_token_feature,  # [n_sc, n_mp, hidden_dim]
            "mp_token_pose": mp_token_pose,  # [n_sc, n_mp, 3]
            "mp_token_type": mp_type,  # [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11
        }
        return mp_tokens
