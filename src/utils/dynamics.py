# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple
from omegaconf import DictConfig
import hydra
from torch.distributions import Independent, Categorical
from utils.transform_utils import cast_rad


class Dynamics:
    def __init__(
        self, veh: DictConfig, ped: DictConfig, cyc: DictConfig, navi_mode: str, use_veh_dynamics_for_all: bool = False
    ) -> None:
        self.dt = 0.1
        self.action_dim = 2
        self.navi_mode = navi_mode
        self.use_veh_dynamics_for_all = use_veh_dynamics_for_all

        if self.use_veh_dynamics_for_all:
            self.ag_dynamics = hydra.utils.instantiate(veh, dt=self.dt)
        else:
            self.ag_dynamics = (
                hydra.utils.instantiate(veh, dt=self.dt),
                hydra.utils.instantiate(ped, dt=self.dt),
                hydra.utils.instantiate(cyc, dt=self.dt),
            )

    def init(
        self,
        tl_state: Tensor,  # [n_sc, n_tl, n_step, tl_state_dim], n_step of tl can be different to agents.
        gt_valid: Tensor,  # [n_sc, n_ag, n_step_hist/n_step], n_step for train, n_step_hist for test/val
        gt_pose: Tensor,  # [n_sc, n_ag, n_step_hist/n_step, 3], (x,y,yaw)
        gt_motion: Tensor,  # [n_sc, n_ag, n_step_hist/n_step, 3], (spd,acc,yaw_rate)
        ag_type: Tensor,  # [n_sc, n_ag, 3]
        ag_attr: Tensor,  # [n_sc, n_ag, ag_attr_dim], type and size
        ag_latent: Optional[Tensor],  # [n_sc, n_ag, hidden_dim]
        ag_latent_valid: Optional[Tensor],  # [n_sc, n_ag] or None
        ag_navi: Optional[Tensor],  # cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag], or dummy None
        ag_navi_valid: Tensor,  # [n_sc, n_ag]
        **kwargs,
    ) -> None:

        self.ag_type, self.ag_attr = ag_type, ag_attr  # constant, [n_sc, n_ag, 3]

        self.ag_latent, self.ag_latent_valid = ag_latent, ag_latent_valid  # constant

        # will be updated in self.disable_ag() and self.override_ag(), [n_sc, n_ag], bool
        self.ag_valid = gt_valid[:, :, 0]
        self.ag_disabled = torch.zeros_like(self.ag_valid)  # disable if out side of map, disabled cannot be re-spawned

        # will be updated in self.update_ag() and self.override_ag()
        self.ag_pose = gt_pose[:, :, 0]  # [n_sc, n_ag, 3], (x,y,yaw)
        self.ag_motion = gt_motion[:, :, 0]  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)

        # will be updated in self.override_tl()
        self.tl_state = tl_state[:, :, 0]  # [n_sc, n_tl, tl_state_dim], bool one_hot

        # will be updated in self.override_navi()
        self.ag_navi = ag_navi
        # will be updated in self.disable_navi(), [n_sc, n_ag], bool
        self.ag_navi_valid = ag_navi_valid  # [n_sc, n_ag]
        self.mask_navi_reached = torch.zeros_like(self.ag_navi_valid)  # model will update the navi by checking this
        self.ag_navi_updated = True  # if True, model will run navi_encoder and set it to False

    def update_ag(
        self, action_dist: Independent, deterministic: bool = True, player_override: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            action_dist: Independent Normal, unbounded value range.
            player_override: None or Dict
                "valid": [n_sc, n_ag]
                "action": [n_sc, n_ag, 2]

        Update attrs:
            self.ag_pose: [n_sc, n_ag, 3], (x,y,yaw)
            self.ag_motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)

        Returns:
            action: [n_sc, n_ag, 2]
            action_log_prob: [n_sc, n_ag]
        """
        mask_type = ~self.ag_type  # [n_sc, n_ag, 3] reversed ag_type for masked_fill
        ag_invalid = ~self.ag_valid.unsqueeze(-1)
        # [n_sc, n_ag, 2], unbounded
        action_unbounded = action_dist.mean if deterministic else action_dist.rsample()

        # [n_sc, n_ag]
        action_log_prob = action_dist.log_prob(action_unbounded.detach()).masked_fill(ag_invalid.squeeze(-1), 0)

        # action: unbounded value to value with physical metrics.
        if self.use_veh_dynamics_for_all:
            action = self.ag_dynamics.process_action(action_unbounded)
        else:
            action = 0
            # mask_type: [n_sc, n_ag, 3]
            for i in range(3):
                action += self.ag_dynamics[i].process_action(action_unbounded).masked_fill(mask_type[:, :, [i]], 0)
        action = action.masked_fill(ag_invalid, 0)

        # override actions: for player-controlled agents.
        if player_override is not None:
            mask_override = (player_override["valid"] & self.ag_valid).unsqueeze(-1)  # [n_sc, n_ag, 1], bool
            action = action.masked_fill(mask_override, 0) + player_override["action"].masked_fill(~mask_override, 0)

        # dynamics update states using actions
        if self.use_veh_dynamics_for_all:
            pred_pose, pred_motion = self.ag_dynamics.update(self.ag_pose, self.ag_motion, action)
        else:
            pred_pose, pred_motion = 0, 0
            for i in range(3):  # mask_type: [n_sc, n_ag, 3]
                _pred_pose, _pred_motion = self.ag_dynamics[i].update(self.ag_pose, self.ag_motion, action)
                pred_pose += _pred_pose.masked_fill(mask_type[:, :, [i]], 0)
                pred_motion += _pred_motion.masked_fill(mask_type[:, :, [i]], 0)

        # mask invalid
        self.ag_pose = pred_pose.masked_fill(ag_invalid, 0)
        self.ag_motion = pred_motion.masked_fill(ag_invalid, 0)
        return action, action_log_prob

    def override_ag(self, ag_override: Dict[str, Tensor]) -> None:
        """For teacher forcing (existing agent) and spawn (new agent).
        Args:
            ag_override: Dict
                "valid": [n_sc, n_ag]
                "pose": [n_sc, n_ag, 3]
                "motion": [n_sc, n_ag, 3]

        Update attrs:
            self.ag_valid: [n_sc, n_ag]
            self.ag_pose: [n_sc, n_ag, 3], (x,y,yaw)
            self.ag_motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        """
        valid = ag_override["valid"] & (~self.ag_disabled)
        if valid.any():
            self.ag_valid = self.ag_valid | valid
            valid = valid.unsqueeze(-1)
            invalid = ~valid
            self.ag_pose = self.ag_pose.masked_fill(valid, 0) + ag_override["pose"].masked_fill(invalid, 0)
            self.ag_motion = self.ag_motion.masked_fill(valid, 0) + ag_override["motion"].masked_fill(invalid, 0)

    @torch.no_grad()
    def override_tl(self, tl_state_dist: Categorical, tl_override: Dict[str, Tensor]) -> None:
        """For teacher forcing (training) or use ground-truth (val/test) traffic lights.
        Args:
            tl_override: Dict
                "valid": [n_sc, n_tl]
                "state": [n_sc, n_tl, tl_state_dim]

        Update attrs:
            self.tl_state: [n_sc, n_tl, tl_state_dim]
        """
        self.tl_state = torch.zeros_like(self.tl_state)
        self.tl_state[
            torch.arange(self.tl_state.shape[0]).unsqueeze(1),
            torch.arange(self.tl_state.shape[1]).unsqueeze(0),
            tl_state_dist.probs.argmax(-1),
        ] = True

        if tl_override["valid"].any():
            valid = tl_override["valid"].unsqueeze(-1)
            self.tl_state = self.tl_state.masked_fill(valid, 0) + tl_override["state"].masked_fill(~valid, 0)

    @torch.no_grad()
    def disable_ag(self, traffic_rule_violation: Dict[str, Tensor], gt_valid: Optional[Tensor] = None) -> None:
        """
        Args:
            traffic_rule_violation: at t
            gt_valid: [n_sc, n_ag] at t

        Update: will take affect at next update(), at t
            self.ag_disabled: [n_sc, n_ag], bool
            self.ag_valid: [n_sc, n_ag], bool
        """
        mask_disable = traffic_rule_violation["outside_map_this_step"]
        if gt_valid is not None:  # do not disable agent that has gt_valid, for training
            mask_disable = mask_disable & (~gt_valid)
        if mask_disable.any():
            self.ag_disabled = self.ag_disabled | mask_disable
            self.ag_valid = self.ag_valid & (~mask_disable)

    @torch.no_grad()
    def disable_navi(self, traffic_rule_violation: Dict[str, Tensor]) -> None:
        """Disable navi once it's reached.
        Args:
            traffic_rule_violation:
                "dest_reached_this_step": [n_sc, n_ag] at t
                "goal_reached_this_step": [n_sc, n_ag] at t
                "cmd_reached_this_step": [n_sc, n_ag] at t, TODO, to be implemented

        Update attr:
            self.ag_navi_valid: [n_sc, n_ag], bool
            self.mask_navi_reached: [n_sc, n_ag], bool
        """
        if self.navi_mode == "dest":
            self.mask_navi_reached = traffic_rule_violation["dest_reached_this_step"]
            self.ag_navi_valid = self.ag_navi_valid & (~self.mask_navi_reached)
        elif self.navi_mode == "goal":
            self.mask_navi_reached = traffic_rule_violation["goal_reached_this_step"]
            self.ag_navi_valid = self.ag_navi_valid & (~self.mask_navi_reached)
        elif self.navi_mode == "cmd":
            # ! cmd_reached check not implemented yet
            self.ag_navi_valid = self.ag_navi_valid

    @torch.no_grad()
    def override_navi(self, navi: Tensor) -> None:
        """Once navi reached, modle might predict a new navi.
        Args:
            navi: cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag], or dummy None
                
        Update attr:
            self.ag_navi: cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag], or dummy None
            self.ag_navi_valid: [n_sc, n_ag], bool
            self.ag_navi_updated
        """
        valid, invalid = self.mask_navi_reached, ~self.mask_navi_reached
        if self.navi_mode in ("cmd", "goal"):
            valid, invalid = valid.unsqueeze(-1), invalid.unsqueeze(-1)
        self.ag_navi = self.ag_navi.masked_fill(valid, 0) + navi.masked_fill(invalid, 0)
        self.ag_navi_valid = self.ag_navi_valid | self.mask_navi_reached
        self.ag_navi_updated = True


class MultiPathPP:
    def __init__(self, dt: float, max_acc: float = 4, max_yaw_rate: float = 1) -> None:
        """
        max_acc:
        max_yaw_rate: veh=1rad/s, cyc=2.5rad/s, ped=5rad/s
        max_yaw_rate: veh=1.2rad/s, cyc=3rad/s, ped=6rad/s
        delta_theta per step (0.1sec), veh: 5m, 0.1rad, cyc: 2m, 0.25rad, ped:1m, 0.5rad
        """
        self.dt = dt
        self._max_acc = max_acc
        self._max_yaw_rate = max_yaw_rate

    def process_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: [n_sc, n_ag, 2] unbounded sample from Gaussian
        Returns:
            action: [n_sc, n_ag, 2], acc (m/s2), theta_rad (rad/s)
        """
        action = torch.tanh(action)
        action = torch.stack([action[..., 0] * self._max_acc, action[..., 1] * self._max_yaw_rate], dim=-1)
        return action

    def update(self, pose: Tensor, motion: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            pose: [n_sc, n_ag, 3], (x,y,yaw)
            motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)
            action: [n_sc, n_ag, 2] vx,vy in m/s

        Returns:
            pred_pose: [n_sc, n_ag, 3], (x,y,yaw)
            pred_motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        """
        acc, yaw_rate = action[:, :, 0], action[:, :, 1]

        # [n_sc, n_ag]
        v_tilde = motion[:, :, 0] + 0.5 * self.dt * acc
        theta_tilde = pose[:, :, 2] + 0.5 * self.dt * yaw_rate
        # [n_sc, n_ag, 3]
        delta_state = torch.stack(
            [v_tilde * torch.cos(theta_tilde), v_tilde * torch.sin(theta_tilde), yaw_rate], dim=-1
        )
        pred_pose = pose + self.dt * delta_state

        # [n_sc, n_ag]
        spd = motion[:, :, 0] + self.dt * acc
        # [n_sc, n_ag, 3]
        pred_motion = torch.stack([spd, acc, yaw_rate], dim=-1)
        return pred_pose, pred_motion


class StateIntegrator:
    def __init__(self, dt: float, max_v: float = 3) -> None:
        self.dt = dt
        self._max_v = max_v  # ped=3m/s

    def process_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: [n_sc, n_ag, 2] unbounded sample from Gaussian
        Returns:
            action: [n_sc, n_ag, 2], vx (m/s), vy (m/s)
        """
        action = torch.tanh(action) * self._max_v
        return action

    def update(self, pose: Tensor, motion: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            pose: [n_sc, n_ag, 3], (x,y,yaw)
            motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)
            action: [n_sc, n_ag, 2] vx,vy in m/s

        Returns:
            pred_pose: [n_sc, n_ag, 3], (x,y,yaw)
            pred_motion: [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        """
        # [n_sc, n_ag]
        vx, vy = action[:, :, 0], action[:, :, 1]
        theta = torch.atan2(vy, vx)  # ? detach

        # [n_sc, n_ag, 3]
        mask_theta = torch.zeros_like(pose, dtype=torch.bool)
        mask_theta[:, :, 2] = True
        pred_pose = pose.masked_fill(mask_theta, 0) + torch.stack([vx * self.dt, vy * self.dt, theta], dim=-1)

        # [n_sc, n_ag]
        spd = torch.norm(action, dim=-1)
        acc = (spd - motion[:, :, 0]) / self.dt
        yaw_rate = cast_rad(theta - pose[:, :, 2]) / self.dt
        # [n_sc, n_ag, 3]
        pred_motion = torch.stack([spd, acc, yaw_rate], dim=-1)
        return pred_pose, pred_motion
