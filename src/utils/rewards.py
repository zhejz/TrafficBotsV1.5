# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from torch import Tensor
import torch
from omegaconf import DictConfig
from models.metrics.loss import AngularError


class DifferentiableReward:
    def __init__(
        self,
        l_pos: DictConfig,
        l_rot: DictConfig,
        l_spd: DictConfig,
        w_collision: float,
        use_il_loss: bool,
        reduce_collsion_with_max: bool,
        is_enabled: bool,
    ):
        # traffic_rule
        self.w_collision = w_collision
        self.reduce_collsion_with_max = reduce_collsion_with_max
        self.is_enabled = is_enabled

        # imitation
        self.use_il_loss = use_il_loss
        if self.use_il_loss:
            self.il_l_pos = getattr(torch.nn, l_pos.criterion)(reduction="none")
            self.il_w_pos = l_pos.weight
            self.il_l_rot = AngularError(l_rot.criterion, l_rot.angular_type)
            self.il_w_rot = l_rot.weight
            self.il_l_spd = getattr(torch.nn, l_spd.criterion)(reduction="none")
            self.il_w_spd = l_spd.weight

    def get(
        self,
        pred_valid: Tensor,  # [n_sc, n_ag], bool
        pred_pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        pred_motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        gt_valid: Tensor,  # [n_sc, n_ag], bool
        gt_pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        gt_motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        ag_size: Tensor,  # [n_sc, n_ag, 3], length, width, height
    ) -> Dict[str, Tensor]:

        if not self.is_enabled:
            return {}

        reward_dict = {
            "diffbar_reward_valid": pred_valid,  # [n_sc, n_ag], bool
            "diffbar_reward": torch.zeros_like(pred_pose[:, :, 0]),  # [n_sc, n_ag]
            "r_imitation_pos": torch.zeros_like(pred_pose[:, :, 0]),  # [n_sc, n_ag]
            "r_imitation_rot": torch.zeros_like(pred_pose[:, :, 0]),  # [n_sc, n_ag]
            "r_imitation_spd": torch.zeros_like(pred_pose[:, :, 0]),  # [n_sc, n_ag]
            "r_traffic_rule_approx": torch.zeros_like(pred_pose[:, :, 0]),  # [n_sc, n_ag]
        }

        if self.use_il_loss and (gt_valid is not None):
            reward_dict["diffbar_reward_valid"] = pred_valid & gt_valid
            error_pos = self.il_l_pos(gt_pose[..., :2], pred_pose[..., :2]).sum(-1)
            error_rot = self.il_l_rot.compute(gt_pose[..., 2], pred_pose[..., 2])
            error_spd = self.il_l_spd(gt_motion[..., 0], pred_motion[..., 0])
            # -1.0 because we call it reward (not penalty)
            reward_dict["r_imitation_pos"] = -1.0 * self.il_w_pos * error_pos
            reward_dict["r_imitation_rot"] = -1.0 * self.il_w_rot * error_rot
            reward_dict["r_imitation_spd"] = -1.0 * self.il_w_spd * error_spd
            _invalid = ~reward_dict["diffbar_reward_valid"]
            reward_dict["r_imitation_pos"].masked_fill_(_invalid, 0.0)
            reward_dict["r_imitation_rot"].masked_fill_(_invalid, 0.0)
            reward_dict["r_imitation_spd"].masked_fill_(_invalid, 0.0)

            reward_dict["diffbar_reward"] = (
                reward_dict["r_imitation_pos"] + reward_dict["r_imitation_rot"] + reward_dict["r_imitation_spd"]
            )

        if self.w_collision > 0:
            reward_dict["diffbar_reward_valid"] = pred_valid
            reward_dict["r_traffic_rule_approx"] = self._get_r_traffic_rule_approx(
                pred_valid, pred_pose, ag_size, self.reduce_collsion_with_max
            )
            reward_dict["r_traffic_rule_approx"] = -1.0 * self.w_collision * reward_dict["r_traffic_rule_approx"]
            reward_dict["r_traffic_rule_approx"].masked_fill_(~reward_dict["diffbar_reward_valid"], 0.0)
            reward_dict["diffbar_reward"] += reward_dict["r_traffic_rule_approx"]

        return reward_dict

    @staticmethod
    def _get_r_traffic_rule_approx(
        pred_valid: Tensor,  # [n_sc, n_ag], bool
        pred_pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        ag_size: Tensor,  # [n_sc, n_ag, 3], length, width, height
        reduce_collsion_with_max: bool,
    ) -> Tensor:

        pred_invalid = ~pred_valid
        n_sc, n_ag = pred_valid.shape
        agent_xy = pred_pose[..., :2]  # [n_sc, n_ag, 2]
        agent_yaw = pred_pose[..., 2]  # [n_sc, n_ag]
        agent_heading = torch.stack([torch.cos(agent_yaw), torch.sin(agent_yaw)], axis=-1)  # [n_sc, n_ag, 2]

        agent_w = ag_size[:, :, :2].amin(-1)  # [n_sc, n_ag]
        agent_l = ag_size[:, :, :2].amax(-1)  # [n_sc, n_ag]
        agent_d = (agent_l - agent_w) / 4.0  # [n_sc, n_ag]
        agent_d = agent_d.unsqueeze(-1).expand(-1, -1, 2)  # [n_sc, n_ag, 2]

        # [n_sc, n_ag, 5, 2] centroid of 5 circles
        centroids = agent_xy.unsqueeze(2).expand(-1, -1, 5, -1) + torch.stack(
            [
                -2 * agent_heading * agent_d,
                -1 * agent_heading * agent_d,
                0 * agent_heading * agent_d,
                1 * agent_heading * agent_d,
                2 * agent_heading * agent_d,
            ],
            dim=2,
        )

        # [n_sc, n_ag, 5, 2] -> [n_sc, n_ag, n_ag, 5, 2]
        centroids_0 = centroids.unsqueeze(2).expand(-1, -1, n_ag, -1, -1)
        centroids_1 = centroids_0.transpose(1, 2)
        # [n_sc, n_ag] -> [n_sc, n_ag, n_ag]
        agent_r = agent_w.unsqueeze(-1).expand(-1, -1, n_ag) / 2.0 + torch.finfo(pred_pose.dtype).eps
        agent_r_sum = agent_r.transpose(1, 2) + agent_r

        distances = torch.zeros([n_sc, n_ag, n_ag, 5, 5], device=pred_pose.device)

        for i in range(5):
            for j in range(5):
                # [n_sc, n_ag, n_ag, 2]
                diff = centroids_0[:, :, :, i, :] - centroids_1[:, :, :, j, :]
                # [n_sc, n_ag, n_ag]
                _dist = torch.norm(diff, dim=-1) + torch.finfo(pred_pose.dtype).eps
                distances[:, :, :, i, j] = _dist

        # [n_sc, n_ag, n_ag]
        distances = torch.min(distances.flatten(start_dim=3, end_dim=4), dim=-1)[0]
        # relaxed collision: 1 for fully overlapped, 0 for not overlapped
        collision = torch.clamp(1 - distances / agent_r_sum, min=0)

        # [n_sc, n_ag, n_ag]
        ego_mask = torch.eye(n_ag, device=pred_pose.device, dtype=torch.bool)[None, :, :].expand(n_sc, -1, -1)
        ego_mask = ego_mask | pred_invalid[:, :, None]
        ego_mask = ego_mask | pred_invalid[:, None, :]
        collision.masked_fill_(ego_mask, 0.0)

        if reduce_collsion_with_max:
            # [n_sc, n_ag, n_ag] -> [n_sc, n_ag]: reduce dim 2
            collision = collision.amax(2)
        else:
            # [n_sc, n_ag, n_ag] -> [n_sc, n_ag, n_ag]
            collision = torch.clamp(collision, max=1)
            # [n_sc, n_ag]: reduce n_ag, as_valid: [n_sc, n_ag]
            collision = collision.sum(-1) / (pred_valid.sum(-1, keepdim=True))
        return collision
