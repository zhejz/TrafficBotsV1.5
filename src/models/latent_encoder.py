# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Optional
from torch import Tensor, nn
import torch
from omegaconf import DictConfig
from copy import deepcopy
from .modules.mlp import MLP
from .modules.distributions import MyDist, DiagGaussian, MultiCategorical
from .agent_encoder import AgentEncoder
from .traffic_light import TrafficLightEncoder
from utils.pose_emb import PoseEmb


class LatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        temporal_down_sample_rate: int,
        share_post_prior_encoders: bool,
        latent_prior: DictConfig,
        latent_post: DictConfig,
        tl_encoder: DictConfig,
        ag_encoder: DictConfig,
        pose_rpe: Optional[PoseEmb],
        time_step_gt: int,
    ):
        super().__init__()
        self.out_dim = latent_dim
        self.dummy = latent_dim <= 0
        self.temporal_down_sample_rate = temporal_down_sample_rate

        if not self.dummy:
            if ag_encoder["temp_window_size"] > 0:
                if self.temporal_down_sample_rate > 1:
                    temp_window_size = (time_step_gt + 1) // self.temporal_down_sample_rate + 1
                else:
                    temp_window_size = time_step_gt + 1
                tl_encoder = deepcopy(tl_encoder)
                ag_encoder = deepcopy(ag_encoder)
                tl_encoder["temp_window_size"] = temp_window_size
                ag_encoder["temp_window_size"] = temp_window_size

            self.tl_encoder_post = TrafficLightEncoder(pose_rpe=pose_rpe, **tl_encoder)
            self.ag_encoder_post = AgentEncoder(pose_rpe=pose_rpe, **ag_encoder)

            if share_post_prior_encoders:
                self.tl_encoder_prior, self.ag_encoder_prior = self.tl_encoder_post, self.ag_encoder_post
            else:
                self.tl_encoder_prior = TrafficLightEncoder(pose_rpe=pose_rpe, **tl_encoder)
                self.ag_encoder_prior = AgentEncoder(pose_rpe=pose_rpe, **ag_encoder)

            # such that prior and posterior distribution have different standard deviation.
            self.latent_dist_prior = DistEncoder(ag_encoder["hidden_dim"], latent_dim, **latent_prior)
            self.latent_dist_post = DistEncoder(ag_encoder["hidden_dim"], latent_dim, **latent_post)

    def forward(
        self,
        ag_valid: Tensor,  # [n_sc, n_ag, n_step]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim]
        ag_motion: Tensor,  # [n_sc, n_ag, n_step, ag_motion_dim]
        ag_pose: Tensor,  # [n_sc, n_ag, n_step, 3]
        ag_type: Tensor,  # [n_sc, n_ag, 3] [Vehicle=0, Pedestrian=1, Cyclist=2] one hot
        tl_state: Tensor,  # [n_sc, n_tl, n_step, tl_state_dim]
        mp_tokens: Dict[str, Tensor],
        tl_tokens: Dict[str, Tensor],
        posterior: bool,
    ) -> Optional[MyDist]:
        """
        Args:
            Dict mp_tokens:
                "mp_token_invalid": [n_sc, n_mp]
                "mp_token_feature": [n_sc, n_mp, hidden_dim]
                "mp_token_pose": [n_sc, n_mp, 3]
                "mp_token_type": [n_sc, n_mp, n_mp_type], one_hot, n_mp_type=11
            Dict tl_tokens:
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
            latent_prior or latent_post: [n_sc, n_ag, latent_dim]
        """
        if self.dummy:
            return None
        elif posterior and self.latent_dist_post.skip_forward:
            return self.latent_dist_post(ag_attr, ag_valid.any(-1), ag_type)
        elif (not posterior) and self.latent_dist_prior.skip_forward:
            return self.latent_dist_prior(ag_attr, ag_valid.any(-1), ag_type)
        else:
            # ! downsampling
            if self.temporal_down_sample_rate > 1:
                assert (ag_valid.shape[-1] - 1) % self.temporal_down_sample_rate == 0
                ag_valid = ag_valid[:, :, :: self.temporal_down_sample_rate]
                ag_motion = ag_motion[:, :, :: self.temporal_down_sample_rate]
                ag_pose = ag_pose[:, :, :: self.temporal_down_sample_rate]
                tl_state = tl_state[:, :, :: self.temporal_down_sample_rate]

            # ! chose posterior or prior networks
            _tl_encoder = self.tl_encoder_post if posterior else self.tl_encoder_prior
            _ag_encoder = self.ag_encoder_post if posterior else self.ag_encoder_prior
            _latent_dist = self.latent_dist_post if posterior else self.latent_dist_prior

            # ! run networks
            tl_token_feature = _tl_encoder(tl_state=tl_state, called_by_latent_encoder=True, **tl_tokens)
            ag_token_feature, _ = _ag_encoder(
                called_by_latent_encoder=True,
                ag_valid=ag_valid,
                ag_attr=ag_attr,
                ag_motion=ag_motion,
                ag_pose=ag_pose,
                tl_token_invalid=tl_tokens["tl_token_invalid"],
                tl_token_feature=tl_token_feature,
                tl_token_pose=tl_tokens["tl_token_pose"],
                **mp_tokens,
            )
            return _latent_dist(ag_token_feature, ag_valid.any(-1), ag_type)


class DistEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        branch_type: bool,
        dist_type: str,  # dist_type in ["std_gaus", "diag_gaus", "cat"]
        mlp_use_layernorm: bool,
        log_std: Optional[float],
        n_cat: int,
        n_layer: int,
    ) -> None:
        super().__init__()
        self.dist_type = dist_type
        self.branch_type = branch_type

        if self.dist_type == "std_gaus":
            self.skip_forward = True
            self.mean = nn.Parameter(torch.zeros(1, 1, out_dim), requires_grad=False)
            self.log_std = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        elif self.dist_type == "diag_gaus":
            self.skip_forward = False
            if self.branch_type:
                self.mlp_mean = nn.ModuleList(
                    [
                        MLP(
                            [hidden_dim] * n_layer + [out_dim],
                            end_layer_activation=False,
                            use_layernorm=mlp_use_layernorm,
                        )
                        for _ in range(3)
                    ]
                )
            else:
                self.mlp_mean = MLP(
                    [hidden_dim] * n_layer + [out_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
                )
            if log_std is None:
                self.log_std = None

                if self.branch_type:
                    self.mlp_log_std = nn.ModuleList(
                        [
                            MLP(
                                [hidden_dim] * n_layer + [out_dim],
                                end_layer_activation=False,
                                use_layernorm=mlp_use_layernorm,
                            )
                            for _ in range(3)
                        ]
                    )
                else:
                    self.mlp_log_std = MLP(
                        [hidden_dim] * n_layer + [out_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
                    )
            else:
                if self.branch_type:
                    self.log_std = nn.ParameterList(
                        [nn.Parameter(log_std * torch.ones(out_dim), requires_grad=True) for _ in range(3)]
                    )
                else:
                    self.log_std = nn.Parameter(log_std * torch.ones(out_dim), requires_grad=True)
        elif self.dist_type == "std_cat":
            self.skip_forward = True
            assert out_dim % n_cat == 0
            self.n_cat = n_cat
            self.n_class = out_dim // self.n_cat
            self.logits = nn.Parameter(torch.zeros(1, 1, self.n_cat, self.n_class), requires_grad=False)
        elif self.dist_type == "cat":
            self.skip_forward = False
            assert out_dim % n_cat == 0
            self.n_cat = n_cat
            self.n_class = out_dim // self.n_cat
            if self.branch_type:
                self.mlp_logits = nn.ModuleList(
                    [
                        MLP(
                            [hidden_dim] * n_layer + [out_dim],
                            end_layer_activation=False,
                            use_layernorm=mlp_use_layernorm,
                        )
                        for _ in range(3)
                    ]
                )
            else:
                self.mlp_logits = MLP(
                    [hidden_dim] * n_layer + [out_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
                )
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, valid: Tensor, ag_type: Tensor) -> MyDist:
        if self.dist_type == "std_gaus":
            out_dist = DiagGaussian(self.mean.expand(*valid.shape, -1), self.log_std, valid=valid)
        elif self.dist_type == "diag_gaus":
            if self.branch_type:
                n_sc, n_ag, n_type = ag_type.shape
                mask_type = ~(ag_type & valid.unsqueeze(-1))  # [n_sc, n_ag, 3]
                mean = 0
                for i in range(n_type):
                    mean += self.mlp_mean[i](x, mask_type[:, :, i])
                log_std = 0
                if self.log_std is None:
                    for i in range(n_type):
                        log_std += self.mlp_log_std[i](x, mask_type[:, :, i])
                else:
                    for i in range(n_type):
                        log_std += (
                            self.log_std[i][None, None, :].expand(n_sc, n_ag, -1).masked_fill(mask_type[:, :, [i]], 0)
                        )
            else:
                invalid = ~valid
                mean = self.mlp_mean(x, invalid)
                log_std = self.mlp_log_std(x, invalid) if self.log_std is None else self.log_std

            out_dist = DiagGaussian(mean, log_std, valid=valid)
        elif self.dist_type == "std_cat":
            out_dist = MultiCategorical(self.logits.expand(*valid.shape, -1, -1), valid=valid)
        elif self.dist_type == "cat":
            if self.branch_type:
                mask_type = ~(ag_type & valid.unsqueeze(-1))  # [n_sc, n_ag, 3]
                logits = 0
                for i in range(mask_type.shape[-1]):
                    logits += self.mlp_logits[i](x, mask_type[:, :, i])
                logits = logits.view(*valid.shape, self.n_cat, self.n_class)
            else:
                logits = self.mlp_logits(x, ~valid).view(*valid.shape, self.n_cat, self.n_class)
            out_dist = MultiCategorical(logits, valid=valid)
        return out_dist
