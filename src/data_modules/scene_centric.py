# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Tuple
from omegaconf import DictConfig
from torch import nn, Tensor
import torch


class SceneCentricPreProcessing(nn.Module):
    def __init__(
        self, time_step_current: int, tl_mode: str, navi_mode: str, dropout_p_history: float, data_size: DictConfig
    ) -> None:
        super().__init__()
        self.n_step_hist = time_step_current + 1
        self.tl_mode = tl_mode
        self.navi_mode = navi_mode
        self.dropout_p_history = dropout_p_history  # [0, 1], turn off if set to negative

        assert self.tl_mode in ["lane", "stop"]
        assert self.navi_mode in ["cmd", "goal", "dest", "dummy"]

        if self.navi_mode == "cmd":
            navi_dim = data_size["agent/cmd"][-1]
        elif self.navi_mode == "goal":
            navi_dim = data_size["agent/goal"][-1]
        else:
            navi_dim = None

        self.model_kwargs = {
            "tl_mode": self.tl_mode,
            "navi_mode": self.navi_mode,
            "navi_dim": navi_dim,
            "n_mp_pl_node": data_size["map/valid"][-1],
            "mp_attr_dim": data_size["map/type"][-1],
            "tl_state_dim": data_size["tl_stop/state"][-1],
            "ag_motion_dim": data_size["agent/spd"][-1] + data_size["agent/acc"][-1] + data_size["agent/yaw_rate"][-1],
            "ag_attr_dim": data_size["agent/size"][-1] + data_size["agent/type"][-1],
        }

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: scene-centric Dict batch, c.f. src/data_modules/data_h5_womd.py

        Returns: Dict, all tensors in scene-centric coordinate
            # (sc) map polylines
                "sc/mp_valid": [n_sc, n_mp, n_mp_pl_node], bool, after random dropout
                "sc/mp_attr": [n_sc, n_mp, mp_attr_dim], lane type
                "sc/mp_pose": [n_sc, n_mp, n_mp_pl_node, 3], (x,y,yaw)
            # (sc) traffic lights: stop point or lane, tracked
                "sc/tl_valid": [n_sc, n_tl]
                "sc/tl_attr": [n_sc, n_tl] idx if tl_lane, or None if tl_stop
                "sc/tl_state": [n_sc, n_tl, n_step_hist, tl_state_dim], bool one_hot
                "sc/tl_pose": [n_sc, n_tl, 3]
            # (sc) agents, tracked
                "sc/ag_valid": [n_sc, n_ag, n_step_hist], bool, after random dropout
                "sc/ag_attr": [n_sc, n_ag, ag_attr_dim], size and type
                "sc/ag_motion": [n_sc, n_ag, n_step_hist, ag_motion_dim], (spd,acc,yaw_rate)
                "sc/ag_pose": [n_sc, n_ag, n_step_hist, 3], (x,y,yaw)
            # (gt) ground-truth for training, not available for testing
                "gt/ag_valid": [n_sc, n_ag, n_step], bool
                "gt/ag_motion": [n_sc, n_ag, n_step, ag_motion_dim], (spd,acc,yaw_rate)
                "gt/ag_pose": [n_sc, n_ag, n_step, 3], (x,y,yaw)
                "gt/ag_navi": cmd [n_sc, n_ag, 8], goal [n_sc, n_ag, 4], dest [n_sc, n_ag]
                "gt/tl_valid": [n_sc, n_tl_stop/n_tl_lane], bool
                "gt/tl_state": [n_sc, n_tl_stop/n_tl_lane, n_step, tl_state_dim], bool one_hot
            # (ref) reference information for warm start and post-processing
                "ref/ag_type": [n_sc, n_ag, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
                "ref/ag_role": [n_sc, n_ag, 3], bool one_hot [sdc=0, interest=1, predict=2]
                "ref/ag_size": [n_sc, n_ag, 3], [length, width, height]
                "ref/mp_type": [n_sc, n_mp, 11]
        """
        prefix = "" if self.training else "history/"

        # ! map, prepare "sc/mp_*"
        batch["sc/mp_valid"] = batch["map/valid"].clone()
        batch["sc/mp_attr"] = batch["map/type"].type_as(batch["map/pos"])
        batch["sc/mp_pose"] = torch.cat(
            [batch["map/pos"][..., :2], torch.atan2(batch["map/dir"][..., [1]], batch["map/dir"][..., [0]])], dim=-1
        )

        # ! traffic lights, prepare "sc/tl_*"
        for k in ("valid", "state"):
            batch[f"sc/tl_{k}"] = batch[f"{prefix}tl_{self.tl_mode}/{k}"][:, :, : self.n_step_hist]
        batch["sc/tl_valid"], batch["sc/tl_state"] = self._merge_invalid_tl_into_state(
            batch["sc/tl_valid"], batch["sc/tl_state"]
        )

        if self.tl_mode == "stop":
            batch["sc/tl_attr"] = None
            batch["sc/tl_pose"] = torch.cat(
                [
                    batch[f"{prefix}tl_stop/pos"][..., :2],
                    torch.atan2(batch[f"{prefix}tl_stop/dir"][..., 1], batch[f"{prefix}tl_stop/dir"][..., 0]),
                ],
                dim=-1,
            )
        elif self.tl_mode == "lane":
            batch["sc/tl_attr"] = batch[f"{prefix}tl_lane/idx"]
            batch["sc/tl_pose"] = batch["sc/mp_pose"][
                torch.arange(batch["sc/mp_pose"].shape[0]).unsqueeze(1), batch["sc/tl_attr"], 0
            ]

        # ! agents, prepare "sc/ag_*"
        batch["sc/ag_valid"] = batch[f"{prefix}agent/valid"][:, :, : self.n_step_hist].clone()
        batch["sc/ag_attr"] = torch.cat([batch[f"{prefix}agent/size"], batch[f"{prefix}agent/type"]], dim=-1)
        batch["sc/ag_motion"] = torch.cat(
            [
                batch[f"{prefix}agent/spd"][:, :, : self.n_step_hist],
                batch[f"{prefix}agent/acc"][:, :, : self.n_step_hist],
                batch[f"{prefix}agent/yaw_rate"][:, :, : self.n_step_hist],
            ],
            dim=-1,
        )
        batch["sc/ag_pose"] = torch.cat(
            [
                batch[f"{prefix}agent/pos"][:, :, : self.n_step_hist, :2],
                batch[f"{prefix}agent/yaw_bbox"][:, :, : self.n_step_hist],
            ],
            dim=-1,
        )

        # ! training/validation time, prepare "gt/*" for losses
        if "agent/valid" in batch.keys():
            batch["gt/ag_valid"] = batch["agent/valid"]
            batch["gt/ag_motion"] = torch.cat([batch["agent/spd"], batch["agent/acc"], batch["agent/yaw_rate"]], dim=-1)
            batch["gt/ag_pose"] = torch.cat([batch["agent/pos"][..., :2], batch["agent/yaw_bbox"]], dim=-1)
            batch["gt/ag_navi"] = None if self.navi_mode == "dummy" else batch[f"agent/{self.navi_mode}"]
            for k in ("valid", "state"):
                batch[f"gt/tl_{k}"] = batch[f"tl_{self.tl_mode}/{k}"]
            batch["gt/tl_valid"], batch["gt/tl_state"] = self._merge_invalid_tl_into_state(
                batch["gt/tl_valid"], batch["gt/tl_state"]
            )

        # ! prepare "ref/*"
        for k in ("type", "role", "size"):
            batch[f"ref/ag_{k}"] = batch[f"{prefix}agent/{k}"]
        batch["ref/mp_type"] = batch["map/type"]

        # ! randomly mask input during training
        if self.training and (0 < self.dropout_p_history <= 1.0):
            # do not mask the first node
            prob_mask = torch.ones_like(batch["sc/mp_valid"][:, :, 1:]) * (1 - self.dropout_p_history)
            batch["sc/mp_valid"][:, :, 1:] &= torch.bernoulli(prob_mask).bool()
            # do not mask the current time step
            prob_mask = torch.ones_like(batch["sc/ag_valid"][..., :-1]) * (1 - self.dropout_p_history)
            batch["sc/ag_valid"][..., :-1] &= torch.bernoulli(prob_mask).bool()

        return batch

    @staticmethod
    def _merge_invalid_tl_into_state(tl_valid: Tensor, tl_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Merge invalid tl to tl_state, LANE_STATE_UNKNOWN = 0;
        Args:
            tl_valid: [n_sc, n_tl, n_step]
            tl_state: [n_sc, n_tl, n_step, n_tl_state]

        Returns:
            tl_valid_any: [n_sc, n_tl], remove invalid traffic lights steps
            tl_state: [n_sc, n_tl, n_step, n_tl_state], invalid tl added to tl_state=0, i.e. UNKNOWN
        """
        tl_valid_any = tl_valid.any(-1)  # [n_sc, n_tl]
        invalid_tl_states = (~tl_valid) & (tl_valid_any.unsqueeze(-1))  # [n_sc, n_tl, n_step]
        tl_state = tl_state | torch.stack(
            [invalid_tl_states] + [torch.zeros_like(invalid_tl_states)] * (tl_state.shape[-1] - 1), dim=-1
        )  # [n_sc, n_tl, n_step, n_tl_state]
        return tl_valid_any, tl_state
