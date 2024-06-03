# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple, Optional, Dict
from torch import Tensor
import torch
from utils.transform_utils import cast_rad


class TeacherForcing:
    def __init__(
        self,
        step_spawn_agent: int = 10,
        step_warm_start: int = 10,
        step_horizon: int = 0,
        step_horizon_decrease_per_epoch: int = 0,
        prob_forcing_agent: float = 0,
        prob_forcing_agent_decrease_per_epoch: float = 0,
        prob_scheduled_sampling: float = 0,
        prob_scheduled_sampling_decrease_per_epoch: float = 0,
        gt_sdc: bool = False,
        threshold_xy: float = -1.0,
        threshold_yaw: float = -1.0,
        threshold_spd: float = -1.0,
    ) -> None:
        """
        Args:
            step_spawn_agent: spawn agents up to this time step
            step_warm_start: teacher forcing all agents up to this time step.
            step_horizon: from step 0 to step_horizon all agents will be teacher-forced
            step_horizon_decrease_per_epoch: decrease `step_horizon` every epoch. e.g. 10
            prob_forcing_agent: some agents will always be teacher-forced. e.g. 0.5
            prob_forcing_agent_decrease_per_epoch: decrease `prob_forcing_agent` every epoch. e.g. 0.1
            prob_scheduled_sampling: apply teacher forcing randomly for each agent at each time step.
            prob_scheduled_sampling_decrease_per_epoch: decrease `prob_scheduled_sampling` every epoch. e.g. 0.1
            gt_sdc: True for what-if motion prediction
            threshold_xy: if > 0, reset based on xy error, turn off by setting to < 0
            threshold_yaw: if > 0, reset based on yaw error, turn off by setting to < 0
            threshold_spd: if > 0, reset based on spd error, turn off by setting to < 0
        """
        self.step_spawn_agent = step_spawn_agent
        self.step_warm_start = step_warm_start
        self.step_horizon = step_horizon
        self.step_horizon_decrease_per_epoch = step_horizon_decrease_per_epoch
        self.prob_forcing_agent = prob_forcing_agent
        self.prob_forcing_agent_decrease_per_epoch = prob_forcing_agent_decrease_per_epoch
        self.prob_scheduled_sampling = prob_scheduled_sampling
        self.prob_scheduled_sampling_decrease_per_epoch = prob_scheduled_sampling_decrease_per_epoch
        self.gt_sdc = gt_sdc
        self.threshold_xy, self.threshold_yaw, self.threshold_spd = threshold_xy, threshold_yaw, threshold_spd

    @torch.no_grad()
    def init(self, ag_valid: Tensor, ag_pose: Tensor, ag_motion: Tensor, tl_state: Tensor, current_epoch: int) -> None:
        """
        Args: ground truth agent and traffic light states, n_step of ag and tl can be different
            ag_valid: [n_sc, n_ag, n_step] bool
            ag_pose: [n_sc, n_ag, n_step, 3], (x,y,yaw)
            ag_motion: [n_sc, n_ag, n_step, 3], (spd,acc,yaw_rate)
            tl_state: [n_sc, n_tl, n_step, tl_state_dim], bool one_hot
            current_epoch: current training epoch

        Set attribute:
            self.tl_teacher_forcing: [n_sc, n_tl, n_step]
            self.ag_teacher_forcing: [n_sc, n_ag, n_step]
        """
        self.ag_valid, self.ag_pose, self.ag_motion, self.tl_state = ag_valid, ag_pose, ag_motion, tl_state
        self.tl_teacher_forcing = torch.ones_like(tl_state[..., 0], dtype=bool)
        self.ag_teacher_forcing = torch.zeros_like(ag_valid)

        # always spawn at step 0
        self.ag_teacher_forcing[:, :, 0] |= ag_valid[:, :, 0]
        if self.step_spawn_agent > 0:
            # spawn when valid change from False to True, because traj is interpolated.
            mask_spawn_agent = (~ag_valid[:, :, :-1]) & ag_valid[:, :, 1:]
            mask_spawn_agent[:, :, self.step_spawn_agent :] = False
            self.ag_teacher_forcing[:, :, 1:] |= mask_spawn_agent

        # warm start
        if self.step_warm_start >= 0:
            self.ag_teacher_forcing[:, :, : self.step_warm_start + 1] |= ag_valid[:, :, : self.step_warm_start + 1]

        # horizon schedule
        step_horizon = int(self.step_horizon - self.step_horizon_decrease_per_epoch * current_epoch)
        if step_horizon > 0:
            self.ag_teacher_forcing[:, :, :step_horizon] |= ag_valid[:, :, :step_horizon]

        # agent schedule
        prob_forcing_agent = self.prob_forcing_agent - self.prob_forcing_agent_decrease_per_epoch * current_epoch
        if prob_forcing_agent > 0:
            # [n_sc, n_ag]
            mask_forcing_agent = torch.bernoulli(torch.ones_like(ag_valid[:, :, 0]) * prob_forcing_agent).bool()
            mask_forcing_agent = mask_forcing_agent.unsqueeze(-1) & ag_valid
            self.ag_teacher_forcing |= mask_forcing_agent

        # scheduled sampling
        prob_scheduled_sampling = (
            self.prob_scheduled_sampling - self.prob_scheduled_sampling_decrease_per_epoch * current_epoch
        )
        if prob_scheduled_sampling > 0:
            # [n_sc, n_ag]
            mask_scheduled_sampling = torch.bernoulli(torch.ones_like(ag_valid) * prob_scheduled_sampling).bool()
            mask_scheduled_sampling &= ag_valid
            self.ag_teacher_forcing |= mask_scheduled_sampling

        # what-if motion prediction
        if self.gt_sdc:
            self.ag_teacher_forcing[:, 0] |= ag_valid[:, 0]

    @torch.no_grad()
    def get(
        self,
        step: int,
        pred_valid: Tensor,  # [n_sc, n_ag], bool
        pred_pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        pred_motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Prepare ag_override and tl_override for rollout

        Attributes:
            self.tl_teacher_forcing: [n_sc, n_tl, n_step], bool
            self.ag_teacher_forcing: [n_sc, n_ag, n_step], bool
            self.ag_pose: [n_sc, n_ag, n_step, 3], (x,y,yaw)
            self.ag_motion: [n_sc, n_ag, n_step, 3], (spd,acc,yaw_rate)
            self.tl_state: [n_sc, n_tl, n_step, tl_state_dim], bool one_hot
            
        Returns:
            ag_override: Dict {"valid", "pose", "motion"} [n_sc, n_ag], [n_sc, n_ag, 3], [n_sc, n_ag, 3]
            tl_override: Dict {"valid", "state"} [n_sc, n_tl], [n_sc, n_tl, tl_state_dim]
        """
        if step > 0 and step < self.ag_teacher_forcing.shape[-1]:
            mask_ag_override = self.ag_teacher_forcing[:, :, step]

            if self.threshold_xy > 0 or self.threshold_yaw > 0 or self.threshold_spd > 0:
                err_invalid = ~(pred_valid & self.ag_valid[:, :, step - 1])  # [n_sc, n_ag], bool

                if self.threshold_xy > 0 or self.threshold_yaw > 0:
                    err_pose = (pred_pose - self.ag_pose[:, :, step - 1]).masked_fill(err_invalid.unsqueeze(-1), 0.0)
                    if self.threshold_xy > 0:
                        mask_ag_override |= torch.norm(err_pose[..., :2], dim=-1) > self.threshold_xy
                    if self.threshold_yaw > 0:
                        mask_ag_override |= torch.abs(torch.rad2deg(cast_rad(err_pose[..., 2]))) > self.threshold_yaw

                if self.threshold_spd > 0:
                    err_spd_m_per_s = torch.abs(
                        (pred_motion[:, :, 0] - self.ag_motion[:, :, step - 1, 0]).masked_fill(err_invalid, 0.0)
                    )
                    mask_ag_override |= err_spd_m_per_s > self.threshold_spd

            ag_override = {
                "valid": mask_ag_override,
                "pose": self.ag_pose[:, :, step],
                "motion": self.ag_motion[:, :, step],
            }
        else:
            ag_override = {
                "valid": torch.zeros_like(self.ag_teacher_forcing[:, :, 0]),
                "pose": torch.zeros_like(self.ag_pose[:, :, 0]),
                "motion": torch.zeros_like(self.ag_motion[:, :, 0]),
            }

        if step > 0 and step < self.tl_teacher_forcing.shape[-1]:
            tl_override = {"valid": self.tl_teacher_forcing[:, :, step], "state": self.tl_state[:, :, step]}
        else:
            tl_override = {
                "valid": torch.zeros_like(self.tl_teacher_forcing[:, :, 0]),
                "state": torch.zeros_like(self.tl_state[:, :, 0]),
            }

        return ag_override, tl_override

