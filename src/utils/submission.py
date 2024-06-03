# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Optional, Dict
from omegaconf import ListConfig
from pathlib import Path
import tarfile
import os
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from waymo_open_dataset.protos import motion_submission_pb2, sim_agents_submission_pb2
from pytorch_lightning.loggers import WandbLogger
from .transform_utils import torch_pos2global, torch_rad2rot


class SubWOMD(Metric):
    def __init__(
        self,
        is_active: bool,
        wb_artifact: str,
        method_name: str,
        authors: ListConfig[str],
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.submission = motion_submission_pb2.MotionChallengeSubmission()
            self.submission.account_name = account_name
            self.submission.unique_method_name = method_name
            self.submission.authors.extend(list(authors))
            self.submission.affiliation = affiliation
            self.submission.description = f"{description}, wb_model: {wb_artifact}"
            self.submission.method_link = method_link
            self.submission.submission_type = 1  # single prediction
            self.submission.uses_lidar_data = False
            self.submission.uses_camera_data = False
            self.submission.uses_public_model_pretraining = False
            self.submission.num_model_parameters = "10M"
            self.submission_scenario_id = []

            self.data_keys = ["trajs", "scores", "object_id", "mask_pred", "scenario_id"]
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx="cat")

    def update(self, batch: Dict[str, Tensor], trajs: Tensor, scores: Tensor) -> None:
        """
        Args:
            trajs: [n_sc, n_ag, K, n_step, 3], (x,y,yaw), downsampled to 2 HZ
            scores: [n_sc, n_ag, K] normalized prob
        """
        trajs = trajs[..., :2]  # [n_sc, n_ag, K, n_step, 2]
        trajs = torch_pos2global(
            trajs.flatten(1, 3), batch["scenario_center"].unsqueeze(1), torch_rad2rot(batch["scenario_yaw"])
        ).view(trajs.shape)

        self.trajs.append(trajs)  # [n_sc, n_ag, K, n_step, 2]
        self.scores.append(scores)  # [n_sc, n_ag, K]
        self.object_id.append(batch["history/agent/object_id"])  # [n_sc, n_ag]
        self.mask_pred.append(batch["ref/ag_role"][..., 2])  # [n_sc, n_ag]

        scenario_id = []
        for str_id in batch["scenario_id"]:
            int_id = [-1] * 16  # max_len of scenario_id string is 16
            for i, c in enumerate(str_id):
                int_id[i] = ord(c)
            scenario_id.append(torch.tensor(int_id, dtype=torch.int32, device=trajs.device).unsqueeze(0))
        self.scenario_id.append(torch.cat(scenario_id, dim=0))  # [n_sc, 16]

    def compute(self) -> Optional[Dict[str, Tensor]]:
        return {k: getattr(self, k) for k in self.data_keys}

    def aggregate_on_cpu(self, gpu_dict_sync: Dict[str, Tensor]) -> None:
        for k in gpu_dict_sync.keys():
            if type(gpu_dict_sync[k]) is list:  # single gpu fix
                gpu_dict_sync[k] = gpu_dict_sync[k][0]

        trajs = gpu_dict_sync["trajs"].cpu().numpy()
        scores = gpu_dict_sync["scores"].cpu().numpy()
        mask_pred = gpu_dict_sync["mask_pred"].cpu().numpy()  # [n_sc, n_ag]
        object_id = gpu_dict_sync["object_id"].cpu().numpy()  # [n_sc, n_ag] int
        scenario_id = []  # list of str
        for ord_id in gpu_dict_sync["scenario_id"]:
            scenario_id.append("".join([chr(x) for x in ord_id if x > 0]))

        n_K = scores.shape[-1]
        for i_batch in range(trajs.shape[0]):
            if scenario_id[i_batch] not in self.submission_scenario_id:
                scenario_prediction = motion_submission_pb2.ChallengeScenarioPredictions()
                scenario_prediction.scenario_id = scenario_id[i_batch]

                agent_pos = trajs[i_batch, mask_pred[i_batch]]  # [n_ag_pred, K, n_step, 2]
                agent_id = object_id[i_batch, mask_pred[i_batch]]  # [n_ag_pred]
                agent_score = scores[i_batch, mask_pred[i_batch]]  # [n_ag_pred, K]

                for i_track in range(agent_pos.shape[0]):
                    prediction = motion_submission_pb2.SingleObjectPrediction()
                    prediction.object_id = agent_id[i_track]
                    for _k in range(n_K):
                        scored_trajectory = motion_submission_pb2.ScoredTrajectory()
                        scored_trajectory.confidence = agent_score[i_track, _k]
                        scored_trajectory.trajectory.center_x.extend(agent_pos[i_track, _k, :, 0])
                        scored_trajectory.trajectory.center_y.extend(agent_pos[i_track, _k, :, 1])
                        prediction.trajectories.append(scored_trajectory)
                    scenario_prediction.single_predictions.predictions.append(prediction)

                self.submission.scenario_predictions.append(scenario_prediction)
                self.submission_scenario_id.append(scenario_id[i_batch])

    def save_sub_file(self, logger: WandbLogger) -> Optional[str]:
        print(f"saving womd submission files to {os.getcwd()}")
        submission_dir = Path(f"{self.submission.unique_method_name}_WOMD")
        submission_dir.mkdir(exist_ok=True)
        f = open(submission_dir / f"{self.submission.unique_method_name}_WOMD.bin", "wb")
        f.write(self.submission.SerializeToString())
        f.close()
        tar_file_name = submission_dir.as_posix() + ".tar.gz"
        with tarfile.open(tar_file_name, "w:gz") as tar:
            tar.add(submission_dir, arcname=submission_dir.name)
        if isinstance(logger, WandbLogger):
            logger.experiment.save(tar_file_name)
        else:
            return tar_file_name


class SubWOSAC(Metric):
    def __init__(
        self,
        is_active: bool,
        wb_artifact: str,
        method_name: str,
        authors: ListConfig[str],
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.wb_artifact = wb_artifact
            self.method_name = method_name
            self.authors = authors
            self.affiliation = affiliation
            self.description = description
            self.method_link = method_link
            self.account_name = account_name
            self.buffer_scenario_rollouts = []
            self.i_file = 0
            self.submission_dir = Path("WOSAC")
            self.submission_dir.mkdir(exist_ok=True)
            self.submission_scenario_id = []

            self.data_keys = [
                "scenario_id",
                "valid_sim",
                "pos_sim",
                "z_sim",
                "yaw_sim",
                "valid_no_sim",
                "object_id_sim",
                "pos_no_sim",
                "z_no_sim",
                "yaw_no_sim",
                "object_id_no_sim",
            ]
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx="cat")

    def update(self, wosac_data: Dict[str, Tensor]) -> None:
        for k in self.data_keys:
            getattr(self, k).append(wosac_data[k])

    def compute(self) -> Optional[Dict[str, Tensor]]:
        return {k: getattr(self, k) for k in self.data_keys}

    def aggregate_on_cpu(self, scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]) -> None:
        for rollout in scenario_rollouts:
            if rollout.scenario_id not in self.submission_scenario_id:
                self.submission_scenario_id.append(rollout.scenario_id)
                self.buffer_scenario_rollouts.append(rollout)
                if len(self.buffer_scenario_rollouts) > 300:
                    self._save_shard()

    def _save_shard(self) -> None:
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.buffer_scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=self.account_name,
            unique_method_name=self.method_name,
            authors=self.authors,
            affiliation=self.affiliation,
            description=f"{self.description}, wb_model: {self.wb_artifact}",
            method_link=self.method_link,
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters="10M",
            acknowledge_complies_with_closed_loop_requirement=True,
        )
        output_filename = self.submission_dir / f"submission.binproto-{self.i_file:05d}"
        print(f"Saving wosac submission files to {output_filename}")
        with open(output_filename, "wb") as f:
            f.write(shard_submission.SerializeToString())
        self.i_file += 1
        self.buffer_scenario_rollouts = []

    def save_sub_file(self, logger: WandbLogger) -> Optional[str]:
        self._save_shard()
        self.i_file = 0
        tar_file_name = self.submission_dir.as_posix() + ".tar.gz"

        print(f"Saving wosac submission files to {tar_file_name}")

        shard_files = sorted([p.as_posix() for p in self.submission_dir.glob("*")])
        with tarfile.open(tar_file_name, "w:gz") as tar:
            for output_filename in shard_files:
                tar.add(output_filename, arcname=output_filename + f"-of-{len(shard_files):05d}")

        if isinstance(logger, WandbLogger):
            logger.experiment.save(tar_file_name)
        else:
            return tar_file_name
