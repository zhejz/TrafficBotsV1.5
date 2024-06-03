# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from utils.transform_utils import cast_rad
from utils.buffer import RolloutBuffer


class ErrorMetrics(Metric):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.add_state("err_counter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_pos_meter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_rot_deg", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_spd_m_per_s", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        buffer: RolloutBuffer,
        gt_valid: Tensor,  # [n_sc, n_ag, n_step], bool
        gt_pose: Tensor,  # [n_sc, n_ag, n_step, 3], (x,y,yaw)
        gt_motion: Tensor,  # [n_sc, n_ag, n_step, 3], (spd,acc,yaw_rate)
    ) -> None:
        """
        Args:
            n_step: ground-truth starts at t=0, ends at t=step_end
            n_step_buffer: buffer starts at t=buffer.step_start, ends at t=buffer.step_end
            buffer.pred_valid: [n_sc, n_joint_future, n_ag, n_step_buffer]
            buffer.pred_pose: [n_sc, n_joint_future, n_ag, n_step_buffer, 3]
            buffer.pred_motion: [n_sc, n_joint_future, n_ag, n_step_buffer, 3]
            buffer.step_start: first step 0 in the rollout buffer corresponds to step_start in absolute time
            buffer.step_end: last step in the rollout buffer corresponds to step_end in absolute time
        """
        gt_valid = gt_valid[:, :, buffer.step_start : buffer.step_end + 1]
        gt_pose = gt_pose[:, :, buffer.step_start : buffer.step_end + 1]
        gt_motion = gt_motion[:, :, buffer.step_start : buffer.step_end + 1]

        err_valid = buffer.pred_valid.squeeze(1) & gt_valid  # [n_sc, n_ag, 1]
        err_invalid = ~err_valid.unsqueeze(-1)  # [n_sc, n_ag, 1, 1]
        # [n_sc, n_ag, n_step, 3]
        err_pose = (buffer.pred_pose.squeeze(1) - gt_pose).masked_fill(err_invalid, 0.0)
        err_motion = (buffer.pred_motion.squeeze(1) - gt_motion).masked_fill(err_invalid, 0.0)

        self.err_counter += err_valid.sum()
        self.err_pos_meter += torch.norm(err_pose[..., :2], dim=-1).sum()
        self.err_rot_deg += torch.abs(torch.rad2deg(cast_rad(err_pose[..., 2]))).sum()
        self.err_spd_m_per_s += torch.abs(err_motion[..., 0]).sum()

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {
            f"{self.prefix}/err/pos_meter": self.err_pos_meter / self.err_counter,
            f"{self.prefix}/err/rot_deg": self.err_rot_deg / self.err_counter,
            f"{self.prefix}/err/spd_m_per_s": self.err_spd_m_per_s / self.err_counter,
        }
        return out_dict


class TrafficRuleMetrics(Metric):
    """
    Log traffic_rule_violations, Not based on ground truth trajectory.
    n_agent_collided / n_agent_valid, collided if collision happened at any time step.
    """

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.add_state("counter_agent", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter_veh", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("outside_map", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("collided", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("run_red_light", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("goal_reached", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dest_reached", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("run_road_edge", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("passive", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, buffer: RolloutBuffer, ag_type: Tensor) -> None:
        """
        Args:
            buffer.pred_valid: [n_sc, n_joint_future, n_ag, n_step]
            buffer.violation: [n_sc, n_joint_future, n_ag, n_step]
            ag_type: [n_sc, n_ag, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
        """
        # [n_sc, n_joint_future, n_ag, n_step]
        valid, invalid = buffer.pred_valid, ~buffer.pred_valid
        outside_map = buffer.violation["outside_map"].masked_fill(invalid, 0)
        collided = buffer.violation["collided"].masked_fill(invalid, 0)
        run_road_edge = buffer.violation["run_road_edge"].masked_fill(invalid, 0)
        run_red_light = buffer.violation["run_red_light"].masked_fill(invalid, 0)
        passive = buffer.violation["passive"].masked_fill(invalid, 0)
        goal_reached = buffer.violation["goal_reached"].masked_fill(invalid, 0)
        dest_reached = buffer.violation["dest_reached"].masked_fill(invalid, 0)

        valid = valid.any(-1)  # [n_sc, n_joint_future, n_ag]
        mask_veh = ag_type[:, None, :, 0]  # [n_sc, 1, n_ag]

        self.counter_agent += valid.sum()
        self.counter_veh += (valid & mask_veh).sum()
        self.outside_map += outside_map.any(-1).sum()
        self.collided += collided.any(-1).sum()
        self.run_road_edge += run_road_edge.any(-1).sum()
        self.run_red_light += run_red_light.any(-1).sum()
        self.passive += passive.any(-1).sum()
        self.goal_reached += goal_reached.any(-1).sum()
        self.dest_reached += dest_reached.any(-1).sum()

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {
            f"{self.prefix}/traffic_rule/outside_map": self.outside_map / self.counter_agent,
            f"{self.prefix}/traffic_rule/collided": self.collided / self.counter_agent,
            f"{self.prefix}/traffic_rule/run_road_edge": self.run_road_edge / self.counter_veh,
            f"{self.prefix}/traffic_rule/run_red_light": self.run_red_light / self.counter_veh,
            f"{self.prefix}/traffic_rule/passive": self.passive / self.counter_veh,
            f"{self.prefix}/traffic_rule/goal_reached": self.goal_reached / self.counter_agent,
            f"{self.prefix}/traffic_rule/dest_reached": self.dest_reached / self.counter_agent,
        }
        return out_dict
