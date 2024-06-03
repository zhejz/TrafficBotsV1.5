# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple, Dict, Optional
from omegaconf import DictConfig
import torch
from torch import Tensor, nn
from torch.distributions import Independent, Categorical
from .map_encoder import MapEncoder
from .traffic_light import TrafficLightEncoder, TrafficLightStatePredictor
from .agent_encoder import AgentEncoder
from .latent_encoder import LatentEncoder
from .navigation import NaviEncoder, NaviPredictor
from .modules.add_navi_latent import AddNaviLatent
from .modules.action_head import ActionHead
from utils.pose_emb import PoseEmb


class TrafficBots(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mp_attr_dim: int,
        tl_state_dim: int,
        ag_attr_dim: int,
        ag_motion_dim: int,
        navi_mode: str,
        navi_dim: int,
        mp_encoder: DictConfig,
        tl_encoder: DictConfig,
        tl_state_predictor: DictConfig,
        ag_encoder: DictConfig,
        navi_encoder: DictConfig,
        navi_predictor: DictConfig,
        latent_encoder: DictConfig,
        tf_cfg: DictConfig,
        time_step_gt: int,
        n_mp_pl_node: int,
        add_navi_latent: DictConfig,
        pose_rpe: DictConfig,
        pairwise_relative: bool,
        temp_window_size: int,
        n_tgt_knn: int,
        dist_limit: float,
        tl_mode: str,
        action_dim: int,
        action_head: DictConfig,
    ) -> None:
        super().__init__()
        self.temp_window_size = temp_window_size

        self.pose_rpe = PoseEmb(pe_dim=hidden_dim, **pose_rpe) if pairwise_relative else None

        self.mp_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            attr_dim=mp_attr_dim,
            n_mp_pl_node=n_mp_pl_node,
            pairwise_relative=pairwise_relative,
            n_tgt_knn=n_tgt_knn,
            dist_limit=dist_limit,
            tf_cfg=tf_cfg,
            pose_rpe=self.pose_rpe,
            **mp_encoder,
        )

        tl_encoder["hidden_dim"] = hidden_dim
        tl_encoder["tl_state_dim"] = tl_state_dim
        tl_encoder["tl_mode"] = tl_mode
        tl_encoder["pairwise_relative"] = pairwise_relative
        tl_encoder["n_tgt_knn"] = n_tgt_knn
        tl_encoder["dist_limit"] = dist_limit
        tl_encoder["tf_cfg"] = tf_cfg
        tl_encoder["temp_window_size"] = temp_window_size
        tl_encoder["temp_encoder"] = mp_encoder["pl_encoder"]
        self.tl_encoder = TrafficLightEncoder(pose_rpe=self.pose_rpe, **tl_encoder)
        self.tl_state_predictor = TrafficLightStatePredictor(
            hidden_dim=hidden_dim, tl_state_dim=tl_state_dim, temp_window_size=temp_window_size, **tl_state_predictor
        )

        ag_encoder["hidden_dim"] = hidden_dim
        ag_encoder["ag_attr_dim"] = ag_attr_dim
        ag_encoder["ag_motion_dim"] = ag_motion_dim
        ag_encoder["pairwise_relative"] = pairwise_relative
        ag_encoder["n_tgt_knn"] = n_tgt_knn
        ag_encoder["dist_limit"] = dist_limit
        ag_encoder["tf_cfg"] = tf_cfg
        ag_encoder["temp_window_size"] = temp_window_size
        ag_encoder["temp_encoder"] = mp_encoder["pl_encoder"]
        self.ag_encoder = AgentEncoder(pose_rpe=self.pose_rpe, **ag_encoder)

        self.latent_encoder = LatentEncoder(
            tl_encoder=tl_encoder,
            ag_encoder=ag_encoder,
            pose_rpe=self.pose_rpe,
            time_step_gt=time_step_gt,
            **latent_encoder,
        )

        self.navi_encoder = NaviEncoder(
            hidden_dim=hidden_dim,
            navi_mode=navi_mode,
            navi_dim=navi_dim,
            pairwise_relative=pairwise_relative,
            mp_pose_emb=self.mp_encoder.pose_emb,
            pose_rpe=self.pose_rpe,
            **navi_encoder,
        )

        self.navi_predictor = NaviPredictor(
            navi_mode=navi_mode, navi_dim=navi_dim, ag_encoder=ag_encoder, pose_rpe=self.pose_rpe, **navi_predictor
        )

        self.add_navi = AddNaviLatent(
            hidden_dim=hidden_dim, in_dim=hidden_dim, dummy=self.navi_encoder.dummy, **add_navi_latent
        )
        self.add_latent = AddNaviLatent(
            hidden_dim=hidden_dim,
            in_dim=self.latent_encoder.out_dim,
            dummy=self.latent_encoder.dummy,
            **add_navi_latent,
        )

        self.action_head = ActionHead(hidden_dim=hidden_dim, action_dim=action_dim, **action_head)

    def _append_hist(self, ag_valid: Tensor, ag_pose: Tensor, ag_motion: Tensor, tl_state: Tensor) -> None:
        ag_valid, ag_pose, ag_motion = ag_valid.unsqueeze(2), ag_pose.unsqueeze(2), ag_motion.unsqueeze(2)
        tl_state = tl_state.unsqueeze(2)
        if self.temp_window_size > 0:
            if self.hist_ag_valid is None:
                self.hist_ag_valid, self.hist_ag_pose, self.hist_ag_motion = ag_valid, ag_pose, ag_motion
                self.hist_tl_state = tl_state
            else:
                self.hist_ag_valid = torch.cat([self.hist_ag_valid, ag_valid], dim=2)
                self.hist_ag_pose = torch.cat([self.hist_ag_pose, ag_pose], dim=2)
                self.hist_ag_motion = torch.cat([self.hist_ag_motion, ag_motion], dim=2)
                self.hist_tl_state = torch.cat([self.hist_tl_state, tl_state], dim=2)

            if self.hist_ag_valid.shape[2] > self.temp_window_size:
                self.hist_ag_valid = self.hist_ag_valid[:, :, -self.temp_window_size :]
                self.hist_ag_pose = self.hist_ag_pose[:, :, -self.temp_window_size :]
                self.hist_ag_motion = self.hist_ag_motion[:, :, -self.temp_window_size :]
                self.hist_tl_state = self.hist_tl_state[:, :, -self.temp_window_size :]
        else:
            self.hist_ag_valid, self.hist_ag_pose, self.hist_ag_motion = ag_valid, ag_pose, ag_motion
            self.hist_tl_state = tl_state

    def init(self) -> None:
        self.hist_ag_valid, self.hist_ag_pose, self.hist_ag_motion, self.hist_tl_state = None, None, None, None
        self.rnn_hidden = None
        self.navi_feature = None
        self.tl_state_predictor.init()

    def forward(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag], bool
        ag_pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        ag_motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_type: Tensor,  # [n_sc, n_ag, 3]
        ag_latent: Optional[Tensor],  # [n_sc, n_ag, hidden_dim]
        ag_latent_valid: Optional[Tensor],  # [n_sc, n_ag]
        ag_navi: Optional[Tensor],  # cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag], or dummy None
        ag_navi_valid: Tensor,  # [n_sc, n_ag], bool
        ag_navi_updated: bool,  # if True, model will run navi_encoder and set it to False
        tl_state: Tensor,  # [n_sc, n_tl, tl_state_dim]
        tl_tokens: Dict[str, Tensor],
        mp_tokens: Dict[str, Tensor],
    ) -> Tuple[Independent, Categorical]:
        """
        Args:
            mp_tokens: pre-computed since map is static
                "mp_token_invalid": [n_sc, n_mp]
                "mp_token_feature": [n_sc, n_mp, hidden_dim]
                "mp_token_pose": [n_sc, n_mp, 3]
            tl_tokens: pre-computed since traffic lights have fixed pose
                "tl_token_invalid": [n_sc, n_tl]
                "tl_token_pose": [n_sc, n_tl, 3]
                "tl_token_attr": [n_sc, n_tl, hidden_dim] or None
                "knn_idx_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl] or None
                "knn_invalid_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl] or None
                "rpe_tl2tl": [n_sc, n_tl, n_tgt_knn_tl2tl, 3] or None
                "knn_tgt_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp] or None
                "knn_invalid_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp] or None
                "rpe_tl2mp": [n_sc, n_tl, n_tgt_knn_tl2mp, 3] or None

        Returns: for each agent a latent distribution that considers temporal relation and interaction between agents.
            action_dist: Independent
            tl_state_dist: Categorical
        """
        self._append_hist(ag_valid, ag_pose, ag_motion, tl_state)

        # ! update navi_feature if navi changed or navi is in local coordinate
        if ag_navi_updated or self.navi_encoder.require_update:
            self.navi_feature = self.navi_encoder(
                ag_navi, ag_pose, mp_tokens["mp_token_feature"], mp_tokens["mp_token_pose"]
            )  # [n_sc, n_ag, hidden_dim] or None

        # ! encode tl feature
        tl_token_feature = self.tl_encoder(tl_state=self.hist_tl_state, **tl_tokens)  # [n_sc, n_tl, hidden_dim]

        # ! encode ag feature
        ag_feature, self.rnn_hidden = self.ag_encoder(
            ag_valid=self.hist_ag_valid,
            ag_attr=ag_attr,
            ag_motion=self.hist_ag_motion,
            ag_pose=self.hist_ag_pose,
            tl_token_invalid=tl_tokens["tl_token_invalid"],
            tl_token_feature=tl_token_feature,
            tl_token_pose=tl_tokens["tl_token_pose"],
            rnn_hidden=self.rnn_hidden,
            **mp_tokens,
        )

        # ! add navi and latent
        ag_feature = self.add_navi(ag_feature, self.navi_feature, ag_navi_valid)
        ag_feature = self.add_latent(ag_feature, ag_latent, ag_latent_valid)

        # ! predict action distribution
        action_dist = self.action_head(ag_feature, ag_valid, ag_type)

        # ! predict tl_state for the next step
        pred_tl_state_logits = self.tl_state_predictor(tl_token_feature, tl_tokens["tl_token_invalid"])
        return action_dist, Categorical(logits=pred_tl_state_logits)
