# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Tuple
import numpy as np
import torch
import tensorflow
from torch import Tensor
from torchmetrics.metric import Metric
from google.protobuf import text_format
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.python.config_util_py import get_breakdown_names_from_motion_config
from waymo_open_dataset.metrics.ops import py_metrics_ops


class WOMDMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self, prefix: str, step_gt: int, step_current: int) -> None:
        """
        submission_type: MOTION_PREDICTION = 1; INTERACTION_PREDICTION = 2; ARGOVERSE2 = 3
        """
        super().__init__()

        self.prefix = prefix
        self.step_gt = step_gt
        self.step_current = step_current
        track_future_samples = step_gt - step_current
        assert track_future_samples == 80
        self.metrics_config, self.metrics_names = self._waymo_metrics_config_names(step_current, track_future_samples)
        self.metrics_config = self.metrics_config.SerializeToString()
        self.metrics_type = ["min_ade", "min_fde", "miss_rate", "overlap_rate", "mean_average_precision"]

        self.m_joint, self.n_pred = 8, 1

        self.cpu_dict = {
            "prediction_trajectory": [],
            "prediction_score": [],
            "ground_truth_trajectory": [],
            "ground_truth_is_valid": [],
            "prediction_ground_truth_indices_mask": [],
            "object_type": [],
        }

        for k in self.cpu_dict.keys():
            self.add_state(k, default=[], dist_reduce_fx="cat")

    def update(self, batch: Dict[str, Tensor], trajs: Tensor, scores: Tensor) -> None:
        """
        Args: pytorch tensors on gpu/cpu
            trajs: [n_sc, n_ag, K, n_step, 3], (x,y,yaw), downsampled to 2 HZ
            scores: [n_sc, n_ag, K] normalized prob
        """
        mask_pred = batch["agent/role"][..., 2]  # [n_sc, n_ag]
        mask_other = (~mask_pred) & batch["agent/valid"][:, :, : self.step_current + 1].all(-1)

        gt_traj = torch.cat(
            [
                batch["agent/pos"][..., :2],  # [n_sc, n_ag, n_step, 2]
                batch["agent/size"][..., :2].unsqueeze(2).expand(-1, -1, batch["agent/pos"].shape[2], -1),
                batch["agent/yaw_bbox"],  # [n_sc, n_ag, n_step, 1]
                batch["agent/vel"],  # [n_sc, n_ag, n_step, 2]
            ],
            axis=-1,
        )
        gt_traj = gt_traj[:, :, : self.step_gt + 1, :]  # [n_sc, n_ag, step_gt, 7]
        gt_valid = batch["agent/valid"][:, :, : self.step_gt + 1]  # [n_sc, n_ag, step_gt]

        # agent_type: [n_sc, n_ag, 3] one_hot -> [n_sc, n_ag] [Vehicle=1, Pedestrian=2, Cyclist=3]
        agent_type = batch["agent/type"].float().argmax(dim=-1) + 1.0

        trajs = trajs[..., :2]  # (x,y,yaw) -> (x,y)
        trajs = trajs.unsqueeze(3)  # [n_sc, n_ag, K, 1, steps, 2]

        n_sc, n_ag, n_step_gt = gt_valid.shape
        n_K, n_step_pred = trajs.shape[2], trajs.shape[4]
        device = trajs.device

        prediction_trajectory = torch.zeros(
            [n_sc, self.m_joint, n_K, self.n_pred, n_step_pred, 2], dtype=torch.float32, device=device
        )
        prediction_score = torch.zeros([n_sc, self.m_joint, n_K], dtype=torch.float32, device=device)
        ground_truth_trajectory = torch.zeros([n_sc, n_ag, n_step_gt, 7], dtype=torch.float32, device=device)
        ground_truth_is_valid = torch.zeros([n_sc, n_ag, n_step_gt], dtype=torch.bool, device=device)
        prediction_ground_truth_indices_mask = torch.zeros(
            [n_sc, self.m_joint, self.n_pred], dtype=torch.bool, device=device
        )
        object_type = torch.zeros([n_sc, n_ag], dtype=torch.float32, device=device)

        for i in range(n_sc):
            # reorder and reduce ground_truth_trajectory and ground_truth_is_valid, first pred_agent then other_agent
            n_pred_agent = mask_pred[i].sum()
            n_other_agent = mask_other[i].sum()

            # trajs: [n_sc, n_ag, K, 1, steps, 2]
            prediction_trajectory[i, :n_pred_agent] = trajs[i, mask_pred[i]]
            prediction_score[i, :n_pred_agent] = scores[i][mask_pred[i]]
            prediction_ground_truth_indices_mask[i, :n_pred_agent] = True

            ground_truth_trajectory[i, :n_pred_agent] = gt_traj[i][mask_pred[i]]
            ground_truth_is_valid[i, :n_pred_agent] = gt_valid[i][mask_pred[i]]
            ground_truth_trajectory[i, n_pred_agent : n_pred_agent + n_other_agent] = gt_traj[i][mask_other[i]]
            ground_truth_is_valid[i, n_pred_agent : n_pred_agent + n_other_agent] = gt_valid[i][mask_other[i]]
            object_type[i, :n_pred_agent] = agent_type[i][mask_pred[i]]
            object_type[i, n_pred_agent : n_pred_agent + n_other_agent] = agent_type[i][mask_other[i]]

        self.prediction_trajectory.append(prediction_trajectory)
        self.prediction_score.append(prediction_score)
        self.ground_truth_trajectory.append(ground_truth_trajectory)
        self.ground_truth_is_valid.append(ground_truth_is_valid)
        self.prediction_ground_truth_indices_mask.append(prediction_ground_truth_indices_mask)
        self.object_type.append(object_type)

    def compute(self) -> Dict[str, Tensor]:
        return {k: getattr(self, k) for k in self.cpu_dict.keys()}

    def aggregate_on_cpu(self, gpu_dict_sync: Dict[str, Tensor]) -> None:
        for k, v in gpu_dict_sync.items():
            if type(v) is list:
                assert len(v) == 1
                v = v[0]
            if v.numel() == 1:
                v = v.unsqueeze(0)
                if k == "prediction_ground_truth_indices_mask":
                    v = v[:, None, None]
            self.cpu_dict[k].append(v.cpu())

    def compute_waymo_motion_metrics(self) -> Dict[str, Tensor]:
        tensorflow.config.set_visible_devices([], "GPU")
        ops_inputs = {}
        for k in self.cpu_dict.keys():
            ops_inputs[k] = torch.cat(self.cpu_dict[k], dim=0)
            self.cpu_dict[k] = []

        indices = torch.arange(self.m_joint, dtype=torch.int64)[None, :, None]  # [1, 8, 1]
        # [n_sc, self.m_joint, self.n_pred]
        ops_inputs["prediction_ground_truth_indices"] = indices.expand(ops_inputs["object_type"].shape[0], -1, -1)

        out_dict = {}
        metric_values = py_metrics_ops.motion_metrics(
            config=self.metrics_config,
            prediction_trajectory=ops_inputs["prediction_trajectory"],
            prediction_score=ops_inputs["prediction_score"],
            ground_truth_trajectory=ops_inputs["ground_truth_trajectory"],
            ground_truth_is_valid=ops_inputs["ground_truth_is_valid"],
            prediction_ground_truth_indices_mask=ops_inputs["prediction_ground_truth_indices_mask"],
            object_type=ops_inputs["object_type"],
            prediction_ground_truth_indices=ops_inputs["prediction_ground_truth_indices"],
        )

        for m_type in self.metrics_type:  # e.g. min_ade
            values = np.array(getattr(metric_values, m_type))
            sum_VEHICLE = 0.0
            sum_PEDESTRIAN = 0.0
            sum_CYCLIST = 0.0
            counter_VEHICLE = 0.0
            counter_PEDESTRIAN = 0.0
            counter_CYCLIST = 0.0
            for i, m_name in enumerate(self.metrics_names):  # e.g. TYPE_CYCLIST_15
                out_dict[f"waymo_metrics/{self.prefix}_{m_type}_{m_name}"] = values[i]
                if "VEHICLE" in m_name:
                    sum_VEHICLE += values[i]
                    counter_VEHICLE += 1
                elif "PEDESTRIAN" in m_name:
                    sum_PEDESTRIAN += values[i]
                    counter_PEDESTRIAN += 1
                elif "CYCLIST" in m_name:
                    sum_CYCLIST += values[i]
                    counter_CYCLIST += 1
            out_dict[f"{self.prefix}/{m_type}"] = values.mean()
            out_dict[f"{self.prefix}/veh/{m_type}"] = sum_VEHICLE / counter_VEHICLE
            out_dict[f"{self.prefix}/ped/{m_type}"] = sum_PEDESTRIAN / counter_PEDESTRIAN
            out_dict[f"{self.prefix}/cyc/{m_type}"] = sum_CYCLIST / counter_CYCLIST
        return out_dict

    @staticmethod
    def _waymo_metrics_config_names(
        track_history_samples: int, track_future_samples: int
    ) -> Tuple[motion_metrics_pb2.MotionMetricsConfig, List[str]]:
        config = motion_metrics_pb2.MotionMetricsConfig()
        config_text = f"""
            track_steps_per_second: 10
            prediction_steps_per_second: 2
            track_history_samples: {track_history_samples}
            track_future_samples: {track_future_samples}
            speed_lower_bound: 1.4
            speed_upper_bound: 11.0
            speed_scale_lower: 0.5
            speed_scale_upper: 1.0
            max_predictions: 6
            """
        config_text += """
            step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
            }"""
        config_text += """
            step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
            }"""
        config_text += """
            step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
            }"""
        text_format.Parse(config_text, config)
        metric_names = get_breakdown_names_from_motion_config(config)
        return config, metric_names
