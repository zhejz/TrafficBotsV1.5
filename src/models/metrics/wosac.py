# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List
from pathlib import Path
from torch import Tensor, tensor
from torchmetrics.metric import Metric
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from waymo_open_dataset.protos import sim_agents_metrics_pb2, sim_agents_submission_pb2, scenario_pb2
import tensorflow
import itertools
import os
import multiprocessing as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self, prefix: str) -> None:
        """
        submission_type: MOTION_PREDICTION = 1; INTERACTION_PREDICTION = 2; ARGOVERSE2 = 3
        """
        super().__init__()
        mp.set_start_method("forkserver", force=True)
        self.prefix = prefix
        self.wosac_config = self.load_metrics_config()

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "min_average_displacement_error",
        ]
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        tensorflow.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(config, scenario_byte, scenario_rollout) -> sim_agents_metrics_pb2.SimAgentMetrics:
        return wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario_pb2.Scenario.FromString(bytes.fromhex(scenario_byte)), scenario_rollout
        )

    def update(self, scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts], scenario_bytes) -> None:
        n_pool = min(len(scenario_rollouts), int(os.getenv("SLURM_CPUS_PER_TASK", len(scenario_rollouts))))
        with mp.Pool(processes=n_pool) as pool:
            pool_scenario_metrics = pool.starmap(
                self._compute_scenario_metrics,
                zip(itertools.repeat(self.wosac_config), scenario_bytes, scenario_rollouts),
            )

        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += scenario_metrics.average_displacement_error
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
            self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
            self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
            self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
            self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
            self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
            self.min_average_displacement_error += scenario_metrics.min_average_displacement_error

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(scenario_id="", **metrics_dict)
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(self.wosac_config, mean_metrics)

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        config_path = Path(wosac_metrics.__file__).parent / "challenge_2024_config.textproto"
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
