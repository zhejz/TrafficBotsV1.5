# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Optional
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from .loss import BalancedKL
from models.modules.distributions import MyDist
from utils.buffer import RolloutBuffer


class TrainingMetrics(Metric):
    def __init__(
        self,
        prefix: str,
        train_navi: bool,
        train_latent: bool,
        w_vae_kl: float,
        kl_balance_scale: float,
        kl_free_nats: float,
        kl_for_unseen_agent: bool,
        w_diffbar_reward: float,
        w_navi: float,
        w_tl_state: float,
        w_relevant_agent: float,
        p_loss_for_irrelevant: float,
        step_training_start: int,
        temporal_discount: float = -1.0,
        loss_for_teacher_forcing: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.p_loss_for_irrelevant = p_loss_for_irrelevant
        self.w_relevant_agent = w_relevant_agent  # set to greater than 0 to enable extra weights on relevant agents
        self.step_training_start = step_training_start
        self.temporal_discount = temporal_discount
        self.loss_for_teacher_forcing = loss_for_teacher_forcing

        # CVAE KL divergence
        self.train_latent = train_latent
        if self.train_latent:
            assert w_vae_kl > 0
            self.w_vae_kl = w_vae_kl
            self.kl_for_unseen_agent = kl_for_unseen_agent
            self.add_state("vae_kl_counter", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("vae_kl", default=tensor(0.0), dist_reduce_fx="sum")
            self.l_vae_kl = BalancedKL(kl_balance_scale=kl_balance_scale, kl_free_nats=kl_free_nats)

        # diffbar reward for state reconstruction (agent x,y,yaw,spd)
        self.w_diffbar_reward = w_diffbar_reward
        self.use_diffbar_reward = self.w_diffbar_reward > 0
        if self.use_diffbar_reward:
            self.add_state("diffbar_reward_counter", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("diffbar_reward", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("dr_il_pos", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("dr_il_rot", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("dr_il_spd", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("dr_rule_apx", default=tensor(0.0), dist_reduce_fx="sum")

        # navigation
        self.train_navi = train_navi
        if self.train_navi:
            assert w_navi > 0
            self.w_navi = w_navi
            self.add_state("navi_loss", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("navi_counter", default=tensor(0.0), dist_reduce_fx="sum")

        # traffic light
        self.w_tl_state = w_tl_state
        self.train_tl_state = self.w_tl_state > 0
        if self.train_tl_state:
            self.add_state("tl_state_loss", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("tl_state_counter", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        buffer: RolloutBuffer,
        ag_role: Tensor,  # [n_sc, n_ag, 3] one_hot [sdc=0, interest=1, predict=2]
        navi_pred: Optional[MyDist],
        navi_gt: Tensor,  # cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag]
        latent_post: Optional[MyDist],
        latent_prior: Optional[MyDist],
    ) -> None:
        """
        buffer:
            pred_valid: [n_sc, n_ag, n_step]
            pred_tl_state: [n_sc, n_tl, n_step, tl_state_dim], logits
            diffbar_reward_valid: [n_sc, n_ag, n_step]
            diffbar_reward: [n_sc, n_ag, n_step]
        """
        with torch.no_grad():
            loss_valid = buffer.pred_valid.squeeze(1)  # [n_sc, n_joint_future, n_ag, n_step] -> [n_sc, n_ag, n_step]
            if self.p_loss_for_irrelevant < 1.0:
                mask_relevant = ag_role.any(-1, keepdim=True)  # [n_sc, n_ag, 1]
                if self.p_loss_for_irrelevant > 0.0:
                    mask_relevant |= torch.bernoulli(torch.ones_like(mask_relevant) * self.p_loss_for_irrelevant).bool()
                loss_valid = loss_valid & mask_relevant
            if self.step_training_start > 0:
                loss_valid = loss_valid.clone()
                loss_valid[:, :, : self.step_training_start] &= False

            if not self.loss_for_teacher_forcing:
                loss_valid &= ~buffer.mask_teacher_forcing.squeeze(1)  # [n_sc, n_ag, n_step]

            if self.w_relevant_agent > 0:
                w_mask_rel = loss_valid.any(-1) + ag_role.any(-1) * self.w_relevant_agent  # [n_sc, n_ag]
            else:
                w_mask_rel = None

        # ! CVAE KL divergence
        if self.train_latent:
            if self.kl_for_unseen_agent:  # posterior valid: unseen agent has unit prior.
                vae_kl_valid = latent_post.valid  # [n_sc, n_ag]
            else:  # prior valid: no kl loss for unseen agent.
                vae_kl_valid = latent_prior.valid  # [n_sc, n_ag]
            vae_kl_valid = vae_kl_valid & (loss_valid.any(-1))
            error_vae = self.l_vae_kl.compute(latent_post.distribution, latent_prior.distribution)
            self.vae_kl_counter += vae_kl_valid.sum()
            if w_mask_rel is not None:
                error_vae *= w_mask_rel
            self.vae_kl += error_vae.masked_fill(~vae_kl_valid, 0.0).sum()

        # ! diffbar reward
        if self.use_diffbar_reward:
            reward_valid = loss_valid & buffer.diffbar_reward["diffbar_reward_valid"].squeeze(1)  # [n_sc, n_ag, n_step]
            error_rewards_dr = buffer.diffbar_reward["diffbar_reward"].squeeze(1)
            error_rewards_dr = error_rewards_dr.masked_fill(~reward_valid, 0.0)  # [n_sc, n_ag, n_step]
            if w_mask_rel is not None:
                error_rewards_dr *= w_mask_rel.unsqueeze(1)

            if self.temporal_discount > 0:
                mask_temp = torch.ones_like(error_rewards_dr)
                for i in range(1, mask_temp.shape[-1]):
                    mask_tf = buffer.mask_teacher_forcing[:, 0, :, i].type(mask_temp.dtype)  # [n_sc, n_ag]
                    mask_temp[:, :, i] = mask_tf + (1 - mask_tf) * mask_temp[:, :, i - 1] * self.temporal_discount
                error_rewards_dr *= mask_temp

            self.diffbar_reward += error_rewards_dr.sum()
            self.diffbar_reward_counter += reward_valid.sum()
            if "r_imitation_pos" in buffer.diffbar_reward:
                self.dr_il_pos += buffer.diffbar_reward["r_imitation_pos"].sum()
                self.dr_il_rot += buffer.diffbar_reward["r_imitation_rot"].sum()
                self.dr_il_spd += buffer.diffbar_reward["r_imitation_spd"].sum()
                self.dr_rule_apx = buffer.diffbar_reward["r_traffic_rule_approx"].sum()

        # ! navi (goal/dest)
        if self.train_navi:
            navi_valid = navi_pred.valid & (loss_valid.any(-1))  # [n_sc, n_ag]
            # same as F.cross_entropy(self.distribution.logits.transpose(1, 2), navi_gt, reduction="none")
            navi_nll = -navi_pred.log_prob(navi_gt).masked_fill(~navi_valid, 0)
            if w_mask_rel is not None:
                navi_nll *= w_mask_rel
            self.navi_loss += navi_nll.sum()
            self.navi_counter += navi_valid.sum()

        # ! traffic light states
        if self.train_tl_state:
            tl_state_nll_valid = ~buffer.tl_state_nll_invalid.squeeze(1)  # [n_sc, n_tl, n_step]
            tl_state_nll = buffer.tl_state_nll.squeeze(1)
            tl_state_nll = tl_state_nll.masked_fill(buffer.tl_state_nll_invalid.squeeze(1), 0.0)
            self.tl_state_loss += tl_state_nll.sum()
            self.tl_state_counter += tl_state_nll_valid.sum()

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {f"{self.prefix}/loss": 0.0}

        if self.train_latent and self.vae_kl_counter > 0:
            out_dict[f"{self.prefix}/vae_kl"] = self.w_vae_kl * self.vae_kl / self.vae_kl_counter
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/vae_kl"]

        if self.use_diffbar_reward and self.diffbar_reward_counter > 0:
            out_dict[f"{self.prefix}/diffbar_reward"] = (
                self.w_diffbar_reward * self.diffbar_reward / self.diffbar_reward_counter
            )
            out_dict[f"{self.prefix}/dr_il_pos"] = self.dr_il_pos / self.diffbar_reward_counter
            out_dict[f"{self.prefix}/dr_il_rot"] = self.dr_il_rot / self.diffbar_reward_counter
            out_dict[f"{self.prefix}/dr_il_spd"] = self.dr_il_spd / self.diffbar_reward_counter
            out_dict[f"{self.prefix}/dr_rule_apx"] = self.dr_rule_apx / self.diffbar_reward_counter

            out_dict[f"{self.prefix}/loss"] -= out_dict[f"{self.prefix}/diffbar_reward"]

        if self.train_navi and self.navi_counter > 0:
            out_dict[f"{self.prefix}/navi_loss"] = self.w_navi * self.navi_loss / self.navi_counter
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/navi_loss"]

        if self.train_tl_state and self.tl_state_counter > 0:
            out_dict[f"{self.prefix}/tl_state_loss"] = self.w_tl_state * self.tl_state_loss / self.tl_state_counter
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/tl_state_loss"]

        return out_dict
