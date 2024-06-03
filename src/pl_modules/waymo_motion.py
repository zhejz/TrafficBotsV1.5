# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Optional, Tuple
import hydra
from pytorch_lightning import LightningModule
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import wandb
from pathlib import Path
from collections import OrderedDict
from models.metrics.training import TrainingMetrics
from models.metrics.logging import ErrorMetrics, TrafficRuleMetrics
from models.modules.distributions import MyDist
from utils.traffic_rule_checker import TrafficRuleChecker
from utils.teacher_forcing import TeacherForcing
from utils.buffer import RolloutBuffer
from utils.rewards import DifferentiableReward
from utils.dynamics import Dynamics
from utils.vis_waymo import VisWaymo
from data_modules.womd_post_processing import WOMDPostProcessing
from data_modules.wosac_post_processing import WOSACPostProcessing
from models.metrics.womd import WOMDMetrics
from utils.submission import SubWOMD, SubWOSAC
from models.metrics.wosac import WOSACMetrics

# torch.autograd.set_detect_anomaly(True)


class WaymoMotion(LightningModule):
    def __init__(
        self,
        time_step_current: int,
        time_step_gt: int,
        time_step_end: int,
        time_step_sim_start: int,
        hidden_dim: int,
        data_size: DictConfig,
        pre_processing: DictConfig,
        model: DictConfig,
        p_training_rollout_prior: float,
        training_detach_model_input: bool,
        training_deterministic_action: bool,
        pred_navi_after_reached: bool,
        differentiable_reward: DictConfig,
        n_vis_batch: int,
        n_joint_future_womd: int,
        n_joint_future_wosac: int,
        joint_future_pred_deterministic_k0: bool,
        womd_post_processing: DictConfig,
        wosac_post_processing: DictConfig,
        dynamics: DictConfig,
        teacher_forcing_training: DictConfig,
        teacher_forcing_reactive_replay: DictConfig,
        teacher_forcing_joint_future_pred: DictConfig,
        training_metrics: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig],
        lr_navi: float,
        sub_womd_reactive_replay: DictConfig,
        sub_womd_joint_future_pred: DictConfig,
        sub_wosac: DictConfig,
        wb_artifact: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # pre_processing
        self.pre_processing = []
        pre_proc_kwargs = {"time_step_gt": time_step_gt}
        for k, v in pre_processing.items():
            _pre_proc = hydra.utils.instantiate(v, time_step_current=time_step_current, data_size=data_size)
            self.pre_processing.append((k, _pre_proc))
            pre_proc_kwargs |= _pre_proc.model_kwargs
        self.pre_processing = nn.Sequential(OrderedDict(self.pre_processing))

        # model
        self.dynamics = Dynamics(navi_mode=pre_proc_kwargs["navi_mode"], **dynamics)
        self.model = hydra.utils.instantiate(
            model, action_dim=self.dynamics.action_dim, **pre_proc_kwargs, _recursive_=False
        )

        # training
        train_navi, train_latent = not self.model.navi_encoder.dummy, not self.model.latent_encoder.dummy
        self.teacher_forcing_training = TeacherForcing(**teacher_forcing_training)
        self.train_metrics_training = TrainingMetrics("training", train_navi, train_latent, **training_metrics)

        # diffbar rewards
        self.diffbar_reward = DifferentiableReward(
            **differentiable_reward, is_enabled=self.train_metrics_training.use_diffbar_reward
        )

        # reactive_replay: scene reconstruct given a complete episode, same setup as training
        self.teacher_forcing_reactive_replay = TeacherForcing(**teacher_forcing_reactive_replay)
        self.train_metrics_reactive_replay = TrainingMetrics(
            "reactive_replay", train_navi, train_latent, **training_metrics
        )
        self.err_metrics_reactive_replay = ErrorMetrics("reactive_replay")
        self.rule_metrics_reactive_replay = TrafficRuleMetrics("reactive_replay")
        self.womd_post_processing = WOMDPostProcessing(
            step_gt=time_step_gt, step_current=time_step_current, **womd_post_processing
        )
        self.womd_metrics_reactive_replay = WOMDMetrics("reactive_replay", time_step_gt, time_step_current)

        # joint_future_pred: no spawn, tl_state from current_step, prior latent, goal/dest
        self.teacher_forcing_joint_future_pred = TeacherForcing(**teacher_forcing_joint_future_pred)
        self.rule_metrics_joint_future_pred = TrafficRuleMetrics("joint_future_pred")
        self.womd_metrics_joint_future_pred = WOMDMetrics("joint_future_pred", time_step_gt, time_step_current)

        # wosac
        self.wosac_post_processing = WOSACPostProcessing(**wosac_post_processing)
        self.wosac_metrics = WOSACMetrics("wosac")

        # save submission files
        self.sub_womd_reactive_replay = SubWOMD(wb_artifact=wb_artifact, **sub_womd_reactive_replay)
        self.sub_womd_joint_future_pred = SubWOMD(wb_artifact=wb_artifact, **sub_womd_joint_future_pred)
        self.sub_wosac = SubWOSAC(wb_artifact=wb_artifact, **sub_wosac)

    def forward(
        self,
        mp_tokens: Dict[str, Tensor],
        tl_tokens: Dict[str, Tensor],
        ag_override: Dict[str, Tensor],
        tl_override: Dict[str, Tensor],
        player_override: Optional[Dict[str, Tensor]] = None,
        deterministic_action: bool = True,
    ) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
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
            ag_override: Dict
                "valid": [n_sc, n_ag]
                "pose": [n_sc, n_ag, 3]
                "motion": [n_sc, n_ag, 3]
            tl_override: Dict
                "valid": [n_sc, n_tl]
                "state": [n_sc, n_tl, tl_state_dim]
            player_override: None or Dict
                "valid": [n_sc, n_ag]
                "action": [n_sc, n_ag, 2]
        Returns:
            pred_dict and vis_dict
        """
        ag_valid, ag_pose, ag_motion = self.dynamics.ag_valid, self.dynamics.ag_pose, self.dynamics.ag_motion
        tl_state = self.dynamics.tl_state
        if self.hparams.training_detach_model_input:
            ag_pose = ag_pose.detach()
            ag_motion = ag_motion.detach()
            tl_state = tl_state.detach()

        action_dist, tl_state_dist = self.model(
            ag_valid=ag_valid,
            ag_pose=ag_pose,
            ag_motion=ag_motion,
            ag_attr=self.dynamics.ag_attr,
            ag_type=self.dynamics.ag_type,
            ag_latent=self.dynamics.ag_latent,
            ag_latent_valid=self.dynamics.ag_latent_valid,
            ag_navi=self.dynamics.ag_navi,
            ag_navi_valid=self.dynamics.ag_navi_valid,
            ag_navi_updated=self.dynamics.ag_navi_updated,
            tl_state=tl_state,
            tl_tokens=tl_tokens,
            mp_tokens=mp_tokens,
        )

        action, action_log_prob = self.dynamics.update_ag(action_dist, deterministic_action, player_override)

        pred_dict = {
            "action_log_prob": action_log_prob,  # [n_sc, n_ag]
            "pred_valid": ag_valid,  # old ag_valid, not updated with ag_override
            "pred_pose": self.dynamics.ag_pose,
            "pred_motion": self.dynamics.ag_motion,
            "pred_tl_state_dist": tl_state_dist,
        }

        self.dynamics.override_ag(ag_override)
        self.dynamics.override_tl(tl_state_dist, tl_override)

        vis_dict = {}
        if not self.training:
            vis_dict = {
                "pred_valid": self.dynamics.ag_valid,  # new ag_valid, updated with ag_override
                "pred_pose": self.dynamics.ag_pose,  # new ag_pose, updated with ag_override
                "pred_motion": self.dynamics.ag_motion,  # new ag_motion, updated with ag_override
                "action": action,
                "ag_navi": self.dynamics.ag_navi,
                "ag_navi_valid": self.dynamics.ag_navi_valid,
                "navi_reached": self.dynamics.mask_navi_reached,
                "tl_state": self.dynamics.tl_state,
            }
        return pred_dict, vis_dict

    def rollout(
        self,
        ag_tokens: Dict[str, Tensor],
        mp_tokens: Dict[str, Tensor],
        tl_tokens: Dict[str, Tensor],
        tl_state_gt: Tensor,  # [n_sc, n_tl, n_step_hist/n_step, tl_state_dim]
        teacher_forcing: TeacherForcing,
        rule_checker: TrafficRuleChecker,
        step_end: int,
        deterministic_action: bool,
        player_policy=None,
    ) -> RolloutBuffer:
        # ! init at step=0
        teacher_forcing.init(
            ag_valid=ag_tokens["gt_valid"],  # [n_sc, n_ag, n_step_hist (train) or n_step (val/test)]
            ag_pose=ag_tokens["gt_pose"],  # [n_sc, n_ag, n_step_hist/n_step, 3], (x,y,yaw)
            ag_motion=ag_tokens["gt_motion"],  # [n_sc, n_ag, n_step_hist/n_step, 3], (spd,acc,yaw_rate)
            tl_state=tl_state_gt,
            current_epoch=self.current_epoch,
        )
        self.dynamics.init(tl_state=tl_state_gt, **ag_tokens)
        self.model.init()
        # player_policy.init()  # todo: dummy player policy

        # ! rollout starting at step=1
        rollout_buffer = RolloutBuffer(step_end, self.hparams.time_step_current)
        rollout_buffer.add_navi_log_prob(ag_tokens["ag_navi_log_prob"], ag_tokens["ag_navi_valid"])
        for _step in range(1, step_end + 1):
            ag_override, tl_override = teacher_forcing.get(
                _step, self.dynamics.ag_valid, self.dynamics.ag_pose, self.dynamics.ag_motion
            )
            player_override = None
            # player_override = player_policy(self.dynamics.ag_pose)  # todo: dummy player policy

            # ! TrafficBots model forward
            pred_dict, vis_dict = self.forward(
                mp_tokens=mp_tokens,
                tl_tokens=tl_tokens,
                ag_override=ag_override,
                tl_override=tl_override,
                player_override=player_override,
                deterministic_action=deterministic_action,
            )
            # ! check traffic rule violation
            violation = rule_checker.check(
                pred_dict["pred_valid"], pred_dict["pred_pose"], pred_dict["pred_motion"], self.dynamics.tl_state
            )
            # ! get gt
            if _step >= ag_tokens["gt_valid"].shape[-1]:
                _gt_valid, _gt_pose, _gt_motion = None, None, None
            else:
                _gt_valid = ag_tokens["gt_valid"][:, :, _step]
                _gt_pose, _gt_motion = ag_tokens["gt_pose"][:, :, _step], ag_tokens["gt_motion"][:, :, _step]

            # ! diffbar_reward for training
            diffbar_reward = self.diffbar_reward.get(
                pred_valid=pred_dict["pred_valid"],
                pred_pose=pred_dict["pred_pose"],
                pred_motion=pred_dict["pred_motion"],
                gt_valid=_gt_valid,
                gt_pose=_gt_pose,
                gt_motion=_gt_motion,
                ag_size=ag_tokens["ag_size"],
            )
            # ! tl_state_nll for training
            if _step >= tl_state_gt.shape[2]:
                tl_state_nll = torch.zeros_like(tl_tokens["tl_token_pose"][:, :, 0])  # [n_sc, n_tl]
                tl_state_nll_invalid = torch.ones_like(tl_tokens["tl_token_invalid"])  # [n_sc, n_tl]
            else:  # tl_state_gt: [n_sc, n_tl, n_step, tl_state_dim]
                _gt_tl_state = tl_state_gt[:, :, _step].max(-1)[1]  # [n_sc, n_tl]
                tl_state_nll = -1.0 * (pred_dict["pred_tl_state_dist"].log_prob(_gt_tl_state))  # [n_sc, n_tl]
                tl_state_nll_invalid = tl_tokens["tl_token_invalid"]  # [n_sc, n_tl]
            # ! adding to buffer
            rollout_buffer.add(
                violation=violation,
                diffbar_reward=diffbar_reward,
                tl_state_nll=tl_state_nll,
                tl_state_nll_invalid=tl_state_nll_invalid,
                vis_dict=vis_dict,
                ag_override=ag_override,
                **pred_dict,
            )
            # ! disable agent and navi
            self.dynamics.disable_ag(violation, _gt_valid)  # disable agent outside of map
            self.dynamics.disable_navi(violation)  # disable navi once reached
            # ! predict navi if any agent reached it's goal
            if self.hparams.pred_navi_after_reached and self.dynamics.mask_navi_reached.any():
                ag_navi_override_dist = self.model.navi_predictor(
                    ag_valid=self.model.hist_ag_valid,
                    ag_attr=self.dynamics.ag_attr,
                    ag_motion=self.model.hist_ag_motion,
                    ag_pose=self.model.hist_ag_pose,
                    ag_type=self.dynamics.ag_type,
                    **mp_tokens,
                )
                if self.model.navi_predictor.navi_mode != "dummy":
                    navi_sample = ag_navi_override_dist.sample(False)
                    ag_navi_log_prob = ag_navi_override_dist.log_prob(navi_sample)
                    rollout_buffer.add_navi_log_prob(ag_navi_log_prob, self.dynamics.mask_navi_reached)
                    self.dynamics.override_navi(navi_sample)
                    rule_checker.update_navi(
                        self.model.navi_predictor.navi_mode, self.dynamics.mask_navi_reached, self.dynamics.ag_navi
                    )

        rollout_buffer.finish()
        return rollout_buffer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! pre_processing
        with torch.no_grad():
            batch = self.pre_processing(batch)
        # ! map
        mp_tokens = self.model.mp_encoder(
            batch["sc/mp_valid"], batch["sc/mp_attr"], batch["sc/mp_pose"], batch["ref/mp_type"]
        )
        # ! traffic light
        tl_tokens = self.model.tl_encoder.pre_compute(
            tl_valid=batch["gt/tl_valid"], tl_attr=batch["sc/tl_attr"], tl_pose=batch["sc/tl_pose"], **mp_tokens
        )
        # ! latent personality
        latent_post = self.model.latent_encoder(
            ag_valid=batch["gt/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["gt/ag_motion"],
            ag_pose=batch["gt/ag_pose"],
            ag_type=batch["ref/ag_type"],
            tl_state=batch["gt/tl_state"],
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            posterior=True,
        )
        latent_prior = self.model.latent_encoder(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            tl_state=batch["sc/tl_state"],
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            posterior=False,
        )
        ag_latent = latent_prior if torch.rand(1) < self.hparams.p_training_rollout_prior else latent_post
        ag_latent_valid = None if ag_latent is None else ag_latent.valid
        ag_latent = None if ag_latent is None else ag_latent.sample(deterministic=False)
        # ! navi
        navi_pred = self.model.navi_predictor(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            **mp_tokens,
        )
        # ! rollout
        rollout_buffer = self.reactive_replay(
            batch=batch,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            ag_latent=ag_latent,
            ag_latent_valid=ag_latent_valid,
            ag_navi=batch["gt/ag_navi"],
            ag_navi_valid=batch["gt/ag_valid"].any(-1),
            teacher_forcing=self.teacher_forcing_training,
            deterministic_action=self.hparams.training_deterministic_action,
        )
        # ! metrics
        metrics_dict = self.train_metrics_training(
            buffer=rollout_buffer,
            ag_role=batch["ref/ag_role"],
            navi_pred=navi_pred,
            navi_gt=batch["gt/ag_navi"],
            latent_post=latent_post,
            latent_prior=latent_prior,
        )
        # ! logging
        for k in metrics_dict.keys():
            self.log(k, metrics_dict[k], on_step=True)
        self.train_metrics_training.reset()
        return metrics_dict[f"{self.train_metrics_training.prefix}/loss"]

    def reactive_replay(
        self,
        batch: Dict[str, Tensor],
        mp_tokens: Dict[str, Tensor],
        tl_tokens: Dict[str, Tensor],
        ag_latent: Optional[Tensor],
        ag_latent_valid: Optional[Tensor],
        ag_navi: Optional[Tensor],
        ag_navi_valid: Tensor,
        teacher_forcing: TeacherForcing,
        deterministic_action: bool,
    ) -> RolloutBuffer:  # reactive_replay works only for train and val dataset (assuming ground truth available)
        rule_checker = TrafficRuleChecker(
            mp_boundary=batch["map/boundary"],
            mp_valid=batch["map/valid"],
            mp_type=batch["map/type"],
            mp_pos=batch["map/pos"],
            mp_dir=batch["map/dir"],
            ag_type=batch["ref/ag_type"],
            ag_size=batch["ref/ag_size"],
            ag_goal=batch["agent/goal"],
            ag_dest=batch["agent/dest"],
            tl_valid=tl_tokens["tl_token_valid"],
            tl_pose=tl_tokens["tl_token_pose"],
            disable_check=self.training,
        )
        ag_tokens = {
            "ag_type": batch["ref/ag_type"],
            "ag_size": batch["ref/ag_size"],
            "ag_attr": batch["sc/ag_attr"],
            "gt_valid": batch["gt/ag_valid"],
            "gt_pose": batch["gt/ag_pose"],
            "gt_motion": batch["gt/ag_motion"],
            "ag_latent": ag_latent,
            "ag_latent_valid": ag_latent_valid,
            "ag_navi": ag_navi,
            "ag_navi_valid": ag_navi_valid,
            "ag_navi_log_prob": torch.zeros_like(batch["sc/ag_attr"][:, :, 0]),  # [n_sc, n_ag]
        }
        rollout_buffer = self.rollout(
            ag_tokens=ag_tokens,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            tl_state_gt=batch["gt/tl_state"],
            teacher_forcing=teacher_forcing,
            rule_checker=rule_checker,
            step_end=self.hparams.time_step_end,
            deterministic_action=deterministic_action,
        )
        rollout_buffer.flatten_joint_future(1)
        return rollout_buffer

    def joint_future_pred(
        self,
        batch: Dict[str, Tensor],
        mp_tokens: Dict[str, Tensor],
        tl_tokens: Dict[str, Tensor],
        ag_latent_dist: Optional[MyDist],
        ag_navi_dist: Optional[MyDist],
        teacher_forcing: TeacherForcing,
        n_joint_future: int,
    ) -> RolloutBuffer:
        # ! repeat_interleave input dict
        ag_tokens = {
            "ag_type": batch["ref/ag_type"],
            "ag_size": batch["ref/ag_size"],
            "ag_attr": batch["sc/ag_attr"],
            "gt_valid": batch["sc/ag_valid"],
            "gt_pose": batch["sc/ag_pose"],
            "gt_motion": batch["sc/ag_motion"],
        }
        ag_tokens = {k: ag_tokens[k].repeat_interleave(n_joint_future, 0) for k in ag_tokens.keys()}
        mp_tokens = {k: mp_tokens[k].repeat_interleave(n_joint_future, 0) for k in mp_tokens.keys()}
        for k in tl_tokens.keys():
            if tl_tokens[k] is not None:
                tl_tokens[k] = tl_tokens[k].repeat_interleave(n_joint_future, 0)
        # ! deterministic navi and latent for sample K0
        if self.hparams.joint_future_pred_deterministic_k0:
            deterministic = torch.zeros_like(ag_tokens["gt_valid"][:, :, 0])  # [n_sc*n_joint_future, n_ag]
            deterministic[::n_joint_future] = True
        else:
            deterministic = False
        # ! sample latent
        if ag_latent_dist is None:
            ag_tokens["ag_latent"], ag_tokens["ag_latent_valid"], ag_latent_log_prob = None, None, None
        else:
            ag_latent_dist.repeat_interleave_(n_joint_future, 0)
            ag_tokens["ag_latent"] = ag_latent_dist.sample(deterministic=deterministic)
            ag_tokens["ag_latent_valid"] = ag_latent_dist.valid
            ag_latent_log_prob = ag_latent_dist.log_prob(ag_tokens["ag_latent"])
            ag_latent_log_prob.masked_fill_(~ag_tokens["ag_latent_valid"], 0)
        # ! sample navi
        _ag_dest = batch["agent/dest"].repeat_interleave(n_joint_future, 0) if "agent/dest" in batch else None
        _ag_goal = batch["agent/goal"].repeat_interleave(n_joint_future, 0) if "agent/goal" in batch else None

        if ag_navi_dist is None:
            ag_tokens["ag_navi"] = None
            ag_tokens["ag_navi_valid"] = torch.zeros_like(ag_tokens["gt_valid"][:, :, 0])  # [n_sc*n_joint_future, n_ag]
            ag_tokens["ag_navi_log_prob"] = torch.zeros_like(batch["sc/ag_attr"][:, :, 0])
        else:
            ag_navi_dist.repeat_interleave_(n_joint_future, 0)
            ag_tokens["ag_navi"] = ag_navi_dist.sample(deterministic)
            ag_tokens["ag_navi_valid"] = ag_navi_dist.valid
            ag_tokens["ag_navi_log_prob"] = ag_navi_dist.log_prob(ag_tokens["ag_navi"])
            ag_tokens["ag_navi_log_prob"].masked_fill_(~ag_tokens["ag_navi_valid"], 0)
            if self.model.navi_predictor.navi_mode == "goal":
                _ag_goal = ag_tokens["ag_navi"]
            elif self.model.navi_predictor.navi_mode == "dest":
                _ag_dest = ag_tokens["ag_navi"]
        # ! init TrafficRuleChecker
        rule_checker = TrafficRuleChecker(
            mp_boundary=batch["map/boundary"].repeat_interleave(n_joint_future, 0),
            mp_valid=batch["map/valid"].repeat_interleave(n_joint_future, 0),
            mp_type=batch["map/type"].repeat_interleave(n_joint_future, 0),
            mp_pos=batch["map/pos"].repeat_interleave(n_joint_future, 0),
            mp_dir=batch["map/dir"].repeat_interleave(n_joint_future, 0),
            ag_type=ag_tokens["ag_type"],
            ag_size=ag_tokens["ag_size"],
            ag_goal=_ag_goal,
            ag_dest=_ag_dest,
            tl_valid=tl_tokens["tl_token_valid"],
            tl_pose=tl_tokens["tl_token_pose"],
            disable_check=self.training,
        )
        # ! rollout
        rollout_buffer = self.rollout(
            ag_tokens=ag_tokens,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            tl_state_gt=batch["sc/tl_state"].repeat_interleave(n_joint_future, 0),
            teacher_forcing=teacher_forcing,
            rule_checker=rule_checker,
            step_end=self.hparams.time_step_end,
            deterministic_action=True,
        )
        rollout_buffer.flatten_joint_future(n_joint_future)
        rollout_buffer.compute_log_prob(ag_latent_log_prob)
        return rollout_buffer

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! pre_processing
        batch = self.pre_processing(batch)
        # ! map
        mp_tokens = self.model.mp_encoder(
            batch["sc/mp_valid"], batch["sc/mp_attr"], batch["sc/mp_pose"], batch["ref/mp_type"]
        )
        # ! traffic light
        tl_tokens = self.model.tl_encoder.pre_compute(
            tl_valid=batch["gt/tl_valid"], tl_attr=batch["sc/tl_attr"], tl_pose=batch["sc/tl_pose"], **mp_tokens
        )
        # ! latent personality
        latent_post = self.model.latent_encoder(
            ag_valid=batch["gt/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["gt/ag_motion"],
            ag_pose=batch["gt/ag_pose"],
            ag_type=batch["ref/ag_type"],
            tl_state=batch["gt/tl_state"],
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            posterior=True,
        )
        latent_prior = self.model.latent_encoder(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            tl_state=batch["sc/tl_state"],
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            posterior=False,
        )
        # ! navi
        navi_pred = self.model.navi_predictor(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            **mp_tokens,
        )
        # ! rollout for reactive_replay: scene reconstruct given a complete episode, same setup as training
        ag_latent = None if latent_post is None else latent_post.sample(deterministic=True)
        ag_latent_valid = None if latent_post is None else latent_post.valid
        buffer_reactive_replay = self.reactive_replay(
            batch=batch,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            ag_latent=ag_latent,
            ag_latent_valid=ag_latent_valid,
            ag_navi=batch["gt/ag_navi"],
            ag_navi_valid=batch["gt/ag_valid"].any(-1),
            teacher_forcing=self.teacher_forcing_reactive_replay,
            deterministic_action=True,
        )
        # ! logging training metrics before joint_future_pred, which will change navi and latent dist inplace
        self.train_metrics_reactive_replay.update(
            buffer=buffer_reactive_replay,
            ag_role=batch["ref/ag_role"],
            navi_pred=navi_pred,
            navi_gt=batch["gt/ag_navi"],
            latent_post=latent_post,
            latent_prior=latent_prior,
        )
        # ! rollout for joint_future_pred: wosac and womd
        buffer_joint_future_pred = self.joint_future_pred(
            batch=batch,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            ag_latent_dist=latent_prior,
            ag_navi_dist=navi_pred,
            teacher_forcing=self.teacher_forcing_joint_future_pred,
            n_joint_future=self.hparams.n_joint_future_wosac,
        )
        # ! logging error metrics, only meaningful for reactive_replay, not for joint_future_pred
        self.err_metrics_reactive_replay.update(
            buffer_reactive_replay, batch["gt/ag_valid"], batch["gt/ag_pose"], batch["gt/ag_motion"]
        )
        # ! logging traffic rule violations
        self.rule_metrics_reactive_replay.update(buffer_reactive_replay, batch["ref/ag_type"])
        self.rule_metrics_joint_future_pred.update(buffer_joint_future_pred, batch["ref/ag_type"])
        # ! WOMD reactive_replay, post_processing, logging metrics, saving submission files
        womd_reactive_replay = self.womd_post_processing(
            ag_type=batch["ref/ag_type"],
            trajs=buffer_reactive_replay.pred_pose[:, :, :, buffer_reactive_replay.step_future_start :],
        )
        # multi gpu womd metrics
        self.womd_metrics_reactive_replay.update(batch, **womd_reactive_replay)
        gpu_dict_sync = self.womd_metrics_reactive_replay.compute()
        if self.global_rank == 0:
            self.womd_metrics_reactive_replay.aggregate_on_cpu(gpu_dict_sync)
        self.womd_metrics_reactive_replay.reset()
        # multi gpu add to sub file
        if self.sub_womd_reactive_replay.is_active:
            self.sub_womd_reactive_replay.update(batch, **womd_reactive_replay)
            gpu_dict_sync = self.sub_womd_reactive_replay.compute()
            if self.global_rank == 0:
                self.sub_womd_reactive_replay.aggregate_on_cpu(gpu_dict_sync)
            self.sub_womd_reactive_replay.reset()

        # ! WOMD joint_future_pred, post_processing, logging metrics, saving submission files
        womd_joint_future_pred = self.womd_post_processing(
            ag_type=batch["ref/ag_type"],
            trajs=buffer_joint_future_pred.pred_pose[:, :, :, buffer_joint_future_pred.step_future_start :],
            scores=buffer_joint_future_pred.log_prob,  # [n_sc, n_joint_pred, n_ag]
        )
        # multi gpu womd metrics
        self.womd_metrics_joint_future_pred.update(batch, **womd_joint_future_pred)
        gpu_dict_sync = self.womd_metrics_joint_future_pred.compute()
        if self.global_rank == 0:
            self.womd_metrics_joint_future_pred.aggregate_on_cpu(gpu_dict_sync)
        self.womd_metrics_joint_future_pred.reset()
        # multi gpu add to sub file
        if self.sub_womd_joint_future_pred.is_active:
            self.sub_womd_joint_future_pred.update(batch, **womd_joint_future_pred)
            gpu_dict_sync = self.sub_womd_joint_future_pred.compute()
            if self.global_rank == 0:
                self.sub_womd_joint_future_pred.aggregate_on_cpu(gpu_dict_sync)
            self.sub_womd_joint_future_pred.reset()

        # ! WOSAC joint_future_pred, post_processing, logging metrics, saving submission files
        wosac_data = self.wosac_post_processing(batch, buffer_joint_future_pred)
        if self.sub_wosac.is_active:
            self.sub_wosac.update(wosac_data)
            gpu_dict_sync = self.sub_wosac.compute()
            if self.global_rank == 0:
                scenario_rollouts = self.wosac_post_processing.get_scenario_rollouts(gpu_dict_sync)
                self.sub_wosac.aggregate_on_cpu(scenario_rollouts)
            self.sub_wosac.reset()
        else:  # wosac api is slow, do not turn it on if we want to save submissions!
            scenario_rollouts = self.wosac_post_processing.get_scenario_rollouts(wosac_data)
            self.wosac_metrics.update(scenario_rollouts, batch["scenario_bytes"])

        # ! visualization
        if self.global_rank == 0 and batch_idx < self.hparams.n_vis_batch:
            self.log_val_video("reactive_replay", batch_idx, batch, buffer_reactive_replay)
            self.log_val_video(
                prefix="joint_future_pred",
                batch_idx=batch_idx,
                batch=batch,
                buffer=buffer_joint_future_pred,
                navi_pred=navi_pred,
                k_to_log=self.hparams.n_joint_future_womd,
            )

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch, on_epoch=True)
        epoch_train_metrics_reactive_replay = self.train_metrics_reactive_replay.compute()
        for k, v in epoch_train_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.train_metrics_reactive_replay.reset()
        epoch_err_metrics_reactive_replay = self.err_metrics_reactive_replay.compute()
        for k, v in epoch_err_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.err_metrics_reactive_replay.reset()
        epoch_rule_metrics_reactive_replay = self.rule_metrics_reactive_replay.compute()
        for k, v in epoch_rule_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.rule_metrics_reactive_replay.reset()
        epoch_rule_metrics_joint_future_pred = self.rule_metrics_joint_future_pred.compute()
        for k, v in epoch_rule_metrics_joint_future_pred.items():
            self.log(k, v, on_epoch=True)
        self.rule_metrics_joint_future_pred.reset()

        if not self.sub_wosac.is_active:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            for k, v in epoch_wosac_metrics.items():
                self.log(k, v, on_epoch=True)
            self.wosac_metrics.reset()

        if self.global_rank == 0:
            epoch_womd_metrics_reactive_replay = self.womd_metrics_reactive_replay.compute_waymo_motion_metrics()
            epoch_womd_metrics_reactive_replay["epoch"] = self.current_epoch
            self.logger[0].experiment.log(epoch_womd_metrics_reactive_replay, commit=True)
            epoch_womd_metrics_joint_future_pred = self.womd_metrics_joint_future_pred.compute_waymo_motion_metrics()
            epoch_womd_metrics_joint_future_pred["epoch"] = self.current_epoch
            self.logger[0].experiment.log(epoch_womd_metrics_joint_future_pred, commit=True)
            if self.sub_womd_reactive_replay.is_active:
                self.sub_womd_reactive_replay.save_sub_file(self.logger[0])
            if self.sub_womd_joint_future_pred.is_active:
                self.sub_womd_joint_future_pred.save_sub_file(self.logger[0])
            if self.sub_wosac.is_active:
                self.sub_wosac.save_sub_file(self.logger[0])

        self.womd_metrics_reactive_replay.reset()
        self.womd_metrics_joint_future_pred.reset()

        self.log("val/loss", epoch_train_metrics_reactive_replay["reactive_replay/loss"], on_epoch=True)

    def log_val_video(
        self,
        prefix: str,
        batch_idx: int,
        batch: Dict[str, Tensor],
        buffer: RolloutBuffer,
        navi_pred: Optional[MyDist] = None,
        k_to_log: int = 1,
        vis_eps_idx: List[int] = [],
    ) -> Tuple[List[str], List[str]]:
        if len(vis_eps_idx) == 0:
            vis_eps_idx = range(buffer.pred_valid.shape[0])
        video_paths = []
        image_paths = []
        scores = torch.zeros_like(buffer.pred_pose[:, :, :, 0, 0]) if buffer.log_prob is None else buffer.log_prob
        scores = scores.softmax(1).cpu().numpy()  # [n_sc, n_joint_future, n_ag]

        for idx in vis_eps_idx:
            video_dir = f"video_{batch_idx}-{idx}"
            _path = Path(video_dir)
            _path.mkdir(exist_ok=True, parents=True)
            map_keys = ["map/valid", "map/type", "map/pos", "map/boundary", "episode_idx"]
            episode = {k: batch[k][idx].cpu().numpy() for k in map_keys if k in batch}
            episode_keys = [
                "agent/valid",
                "agent/pos",
                "agent/yaw_bbox",
                "agent/spd",
                "agent/role",
                "agent/size",
                "tl_lane/valid",
                "tl_lane/state",
                "tl_lane/idx",
                "tl_stop/valid",
                "tl_stop/state",
                "tl_stop/pos",
                "tl_stop/dir",
            ]
            if "agent/valid" in batch:  # train/val
                pf = ""
                episode["agent/goal"] = batch["agent/goal"][idx].cpu().numpy()
                episode["agent/dest"] = batch["agent/dest"][idx].cpu().numpy()
            else:  # test
                pf = "history/"
            for k in episode_keys:
                episode[k] = batch[pf + k][idx].cpu().numpy()
            for k in ["map/pos", "agent/pos", "tl_stop/pos", "tl_stop/dir"]:
                episode[k] = episode[k][..., :2]  # (x,y,z)->(x,y)

            for kf in range(k_to_log):
                prediction = {
                    # [n_ag, n_step_future]
                    "agent/valid": buffer.vis_dict["pred_valid"][idx, kf, :, buffer.step_future_start :],
                    # [n_ag, n_step_future, 2]
                    "agent/pos": buffer.vis_dict["pred_pose"][idx, kf, :, buffer.step_future_start :, :2],
                    # [n_ag, n_step_future, 1]
                    "agent/yaw_bbox": buffer.vis_dict["pred_pose"][idx, kf, :, buffer.step_future_start :, [2]],
                    # [n_ag, n_step_future, 3], (spd,acc,yaw_rate)
                    "motion": buffer.vis_dict["pred_motion"][idx, kf, :, buffer.step_future_start :],
                    "act_P": buffer.action_log_prob[idx, kf, :, buffer.step_future_start :].float().exp(),
                    "ag_navi_valid": buffer.vis_dict["ag_navi_valid"][idx, kf, :, buffer.step_future_start :],
                    "navi_reached": buffer.vis_dict["navi_reached"][idx, kf, :, buffer.step_future_start :],
                    # [n_ag, n_step_future, 2]
                    "action": buffer.vis_dict["action"][idx, kf, :, buffer.step_future_start :],
                }
                for k_dr, v_dr in buffer.diffbar_reward.items():
                    prediction[k_dr] = v_dr[idx, kf, :, buffer.step_future_start :]
                for k in buffer.violation.keys():  # [n_step, n_ag]
                    prediction[k] = buffer.violation[k][idx, kf, :, buffer.step_future_start :]
                if self.model.navi_predictor.navi_mode == "goal":
                    prediction["agent/goal"] = buffer.vis_dict["ag_navi"][idx, kf, :, buffer.step_future_start :]
                elif self.model.navi_predictor.navi_mode == "dest":
                    prediction["agent/dest"] = buffer.vis_dict["ag_navi"][idx, kf, :, buffer.step_future_start :]
                if self.model.tl_encoder.tl_mode == "stop":
                    prediction["tl_stop/state"] = buffer.vis_dict["tl_state"][idx, kf, :, buffer.step_future_start :]
                elif self.model.tl_encoder.tl_mode == "lane":
                    prediction["tl_lane/state"] = buffer.vis_dict["tl_state"][idx, kf, :, buffer.step_future_start :]

                prediction = {k: prediction[k].cpu().numpy() for k in prediction.keys()}
                prediction["step_current"] = self.hparams.time_step_current
                prediction["step_gt"] = self.hparams.time_step_gt if pf == "" else self.hparams.time_step_current
                prediction["step_end"] = self.hparams.time_step_end
                prediction["score"] = scores[idx, kf]  # [n_ag]

                vis_waymo = VisWaymo(
                    episode["map/valid"], episode["map/type"], episode["map/pos"], episode["map/boundary"]
                )
                video_paths_pred = vis_waymo.save_prediction_videos(f"{video_dir}/{prefix}_K{kf}", episode, prediction)
                video_paths += video_paths_pred

                if (navi_pred is not None) and (kf == 0):
                    dest_img_paths = vis_waymo.get_dest_prob_image(
                        f"{video_dir}/dest_im", episode, navi_pred.probs[idx].cpu().numpy()
                    )
                    image_paths += dest_img_paths

        if self.logger is not None:
            for v_p in video_paths:
                self.logger[0].experiment.log({v_p: wandb.Video(v_p)}, commit=False)
            for i_p in image_paths:
                self.logger[0].experiment.log({i_p: wandb.Image(i_p)}, commit=False)
        return video_paths, image_paths

    def configure_optimizers(self):
        params = []
        params_navi = []
        for k, v in self.named_parameters():
            if "navi_predictor" in k:
                params_navi.append(v)
            else:
                params.append(v)
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=params)
        if len(params_navi) > 0:
            optimizer.add_param_group({"params": params_navi, "lr": self.hparams.lr_navi})
        scheduler = {
            "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! pre_processing
        batch = self.pre_processing(batch)
        # ! map
        mp_tokens = self.model.mp_encoder(
            batch["sc/mp_valid"], batch["sc/mp_attr"], batch["sc/mp_pose"], batch["ref/mp_type"]
        )
        # ! traffic light
        tl_tokens = self.model.tl_encoder.pre_compute(
            tl_valid=batch["sc/tl_valid"], tl_attr=batch["sc/tl_attr"], tl_pose=batch["sc/tl_pose"], **mp_tokens
        )
        # ! latent personality
        latent_prior = self.model.latent_encoder(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            tl_state=batch["sc/tl_state"],
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            posterior=False,
        )
        # ! navi
        navi_pred = self.model.navi_predictor(
            ag_valid=batch["sc/ag_valid"],
            ag_attr=batch["sc/ag_attr"],
            ag_motion=batch["sc/ag_motion"],
            ag_pose=batch["sc/ag_pose"],
            ag_type=batch["ref/ag_type"],
            **mp_tokens,
        )
        # ! rollout for joint_future_pred: wosac and womd
        buffer_joint_future_pred = self.joint_future_pred(
            batch=batch,
            mp_tokens=mp_tokens,
            tl_tokens=tl_tokens,
            ag_latent_dist=latent_prior,
            ag_navi_dist=navi_pred,
            teacher_forcing=self.teacher_forcing_joint_future_pred,
            n_joint_future=self.hparams.n_joint_future_wosac,
        )
        # ! logging traffic rule violations
        self.rule_metrics_joint_future_pred.update(buffer_joint_future_pred, batch["ref/ag_type"])
        # ! WOMD joint_future_pred, post_processing, saving submission files
        womd_joint_future_pred = self.womd_post_processing(
            ag_type=batch["ref/ag_type"],
            trajs=buffer_joint_future_pred.pred_pose[:, :, :, buffer_joint_future_pred.step_future_start :],
            scores=buffer_joint_future_pred.log_prob,  # [n_sc, n_joint_pred, n_ag]
        )
        # multi gpu add to sub file
        if self.sub_womd_joint_future_pred.is_active:
            self.sub_womd_joint_future_pred.update(batch, **womd_joint_future_pred)
            gpu_dict_sync = self.sub_womd_joint_future_pred.compute()
            if self.global_rank == 0:
                self.sub_womd_joint_future_pred.aggregate_on_cpu(gpu_dict_sync)
            self.sub_womd_joint_future_pred.reset()

        # ! WOSAC joint_future_pred, post_processing, saving submission files
        wosac_data = self.wosac_post_processing(batch, buffer_joint_future_pred)
        if self.sub_wosac.is_active:
            self.sub_wosac.update(wosac_data)
            gpu_dict_sync = self.sub_wosac.compute()
            if self.global_rank == 0:
                scenario_rollouts = self.wosac_post_processing.get_scenario_rollouts(gpu_dict_sync)
                self.sub_wosac.aggregate_on_cpu(scenario_rollouts)
            self.sub_wosac.reset()

        # ! visualization
        if self.global_rank == 0 and batch_idx < self.hparams.n_vis_batch:
            self.log_val_video(
                prefix="joint_future_pred",
                batch_idx=batch_idx,
                batch=batch,
                buffer=buffer_joint_future_pred,
                navi_pred=navi_pred,
                k_to_log=self.hparams.n_joint_future_womd,
            )

    def test_epoch_end(self, outputs):
        epoch_rule_metrics_joint_future_pred = self.rule_metrics_joint_future_pred.compute()
        for k, v in epoch_rule_metrics_joint_future_pred.items():
            self.log(k, v, on_epoch=True)
        self.rule_metrics_joint_future_pred.reset()
        if self.global_rank == 0:
            if self.sub_womd_joint_future_pred.is_active:
                self.sub_womd_joint_future_pred.save_sub_file(self.logger[0])
            if self.sub_wosac.is_active:
                self.sub_wosac.save_sub_file(self.logger[0])
