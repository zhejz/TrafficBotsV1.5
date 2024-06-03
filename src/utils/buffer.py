# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Optional
from torch import Tensor
import torch


class RolloutBuffer:
    def __init__(self, step_end: int, step_current: int) -> None:
        """
        saves prediction, step [self.step_start,...,self.step_end] in absolute time
        """
        self.step_start = 1  # first step 0 in the rollout buffer corresponds to step_start=1 in absolute time
        self.step_end = step_end  # last step in the rollout buffer corresponds to step_end in absolute time
        # self.step_future_start in the rollout buffer corresponds to step_current+1 in absolute time
        self.step_future_start = step_current

        self.pred_valid: List[Tensor] = []  # n_step * [n_sc, n_ag], bool
        self.pred_pose: List[Tensor] = []  # n_step * [n_sc, n_ag, 3], with grad, (x,y,yaw)
        self.pred_motion: List[Tensor] = []  # n_step * [n_sc, n_ag, 3], with grad, (spd,acc,yaw_rate)
        self.action_log_prob: List[Tensor] = []  # n_step * [n_sc, n_ag]
        self.navi_log_prob: List[Tensor] = []  # n_step_navi_update * [n_sc, n_ag]
        self.navi_log_prob_valid: List[Tensor] = []  # n_step_navi_update * [n_sc, n_ag]

        self.tl_state_nll: List[Tensor] = []  # n_step * [n_sc, n_tl], with grad
        self.tl_state_nll_invalid: List[Tensor] = []  # n_step * [n_sc, n_tl]

        self.diffbar_reward = {}

        self.mask_teacher_forcing: List[Tensor] = []  # n_step * [n_sc, n_ag], bool

        # for simulation metrics
        self.violation = {}  # Dict: n_step * [n_sc, n_ag], no_grad

        #  for visualizing video
        self.vis_dict = {}

        self.log_prob = None

    def add(
        self,
        violation: Dict[str, Tensor],
        diffbar_reward: Dict[str, Tensor],  # [n_sc, n_ag]
        tl_state_nll: Tensor,  # [n_sc, n_tl]
        tl_state_nll_invalid: Tensor,  # [n_sc, n_tl]
        vis_dict: Dict[str, Tensor],
        pred_valid: Tensor,  # [n_sc, n_ag]
        pred_pose: Tensor,  # [n_sc, n_ag, 3]
        pred_motion: Tensor,  # [n_sc, n_ag, 3]
        action_log_prob: Tensor,  # [n_sc, n_ag]
        ag_override: Dict[str, Tensor],
        **kwargs,
    ) -> None:

        self.pred_valid.append(pred_valid)
        self.pred_pose.append(pred_pose)
        self.pred_motion.append(pred_motion)
        self.tl_state_nll.append(tl_state_nll)
        self.tl_state_nll_invalid.append(tl_state_nll_invalid)

        if len(self.violation) == 0:
            self.violation = {k: [] for k in violation.keys()}
        for k, v in violation.items():
            self.violation[k].append(v)

        if len(self.diffbar_reward) == 0:
            self.diffbar_reward = {k: [] for k in diffbar_reward.keys()}
        for k, v in diffbar_reward.items():
            self.diffbar_reward[k].append(v)

        self.action_log_prob.append(action_log_prob)

        if len(self.vis_dict) == 0:
            self.vis_dict = {k: [] for k in vis_dict.keys()}
        for k, v in vis_dict.items():
            if v is not None:
                self.vis_dict[k].append(v)

        self.mask_teacher_forcing.append(ag_override["valid"])

    def finish(self) -> None:
        self.pred_valid = torch.stack(self.pred_valid, dim=2)  # [n_sc, n_ag, n_step]
        self.pred_pose = torch.stack(self.pred_pose, dim=2)  # [n_sc, n_ag, n_step, 3]
        self.pred_motion = torch.stack(self.pred_motion, dim=2)  # [n_sc, n_ag, n_step, 3]
        self.tl_state_nll = torch.stack(self.tl_state_nll, dim=2)  # [n_sc, n_tl, n_step]
        self.tl_state_nll_invalid = torch.stack(self.tl_state_nll_invalid, dim=2)  # [n_sc, n_tl, n_step]
        self.navi_log_prob = torch.stack(self.navi_log_prob, dim=2)  # [n_sc, n_ag, n_step_navi_update]
        self.navi_log_prob_valid = torch.stack(self.navi_log_prob_valid, dim=2)  # [n_sc, n_ag, n_step_navi_update]

        for k in self.violation.keys():  # [n_sc, n_ag, n_step], no_grad
            self.violation[k] = torch.stack(self.violation[k], dim=2)

        for k in self.diffbar_reward.keys():  # [n_sc, n_ag, n_step], grad
            self.diffbar_reward[k] = torch.stack(self.diffbar_reward[k], dim=2)

        self.action_log_prob = torch.stack(self.action_log_prob, dim=2)  # [n_sc, n_ag, n_step]

        for k in self.vis_dict.keys():
            if len(self.vis_dict[k]) > 0:
                self.vis_dict[k] = torch.stack(self.vis_dict[k], dim=2)

        self.mask_teacher_forcing = torch.stack(self.mask_teacher_forcing, dim=2)

    def add_navi_log_prob(self, ag_navi_log_prob: Tensor, mask_navi_reached: Tensor) -> None:
        self.navi_log_prob.append(ag_navi_log_prob)
        self.navi_log_prob_valid.append(mask_navi_reached)

    def compute_log_prob(self, latent_log_prob: Optional[Tensor]) -> None:
        self.log_prob = (self.navi_log_prob * self.navi_log_prob_valid).sum(-1)  # [n_sc, n_joint_future, n_ag]
        self.log_prob /= self.navi_log_prob_valid.sum(-1)
        valid = self.navi_log_prob_valid.any(-1)
        self.log_prob.masked_fill_(~valid, 0)
        if latent_log_prob is not None:
            self.log_prob = self.log_prob + latent_log_prob.view(self.log_prob.shape)

    def flatten_joint_future(self, n_joint_future: int) -> None:
        n_sc, n_ag, n_step = self.pred_valid.shape
        n_sc = n_sc // n_joint_future

        self.pred_valid = self.pred_valid.view(n_sc, n_joint_future, n_ag, n_step)
        self.pred_pose = self.pred_pose.view(n_sc, n_joint_future, n_ag, n_step, -1)
        self.pred_motion = self.pred_motion.view(n_sc, n_joint_future, n_ag, n_step, -1)

        self.navi_log_prob = self.navi_log_prob.view(n_sc, n_joint_future, n_ag, -1)
        self.navi_log_prob_valid = self.navi_log_prob_valid.view(n_sc, n_joint_future, n_ag, -1)

        n_tl = self.tl_state_nll.shape[1]
        self.tl_state_nll = self.tl_state_nll.view(n_sc, n_joint_future, n_tl, n_step)
        self.tl_state_nll_invalid = self.tl_state_nll_invalid.view(n_sc, n_joint_future, n_tl, n_step)

        for k in self.violation.keys():
            self.violation[k] = self.violation[k].view(n_sc, n_joint_future, n_ag, n_step)

        for k in self.diffbar_reward.keys():
            self.diffbar_reward[k] = self.diffbar_reward[k].view(
                *([n_sc, n_joint_future] + list(self.diffbar_reward[k].shape[1:]))
            )

        self.action_log_prob = self.action_log_prob.view(n_sc, n_joint_future, n_ag, n_step)

        for k in self.vis_dict.keys():
            if len(self.vis_dict[k]) > 0:
                self.vis_dict[k] = self.vis_dict[k].view(*([n_sc, n_joint_future] + list(self.vis_dict[k].shape[1:])))

        self.mask_teacher_forcing = self.mask_teacher_forcing.view(
            *([n_sc, n_joint_future] + list(self.mask_teacher_forcing.shape[1:]))
        )
