# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List
import numpy as np
import torch
from torch import nn, Tensor
from waymo_open_dataset.protos import sim_agents_submission_pb2
from utils.transform_utils import torch_pos2global, torch_rad2rot, torch_rad2global
from waymo_open_dataset.utils.sim_agents import submission_specs
from utils.buffer import RolloutBuffer


class WOSACPostProcessing(nn.Module):
    def __init__(
        self,
        step_gt: int,
        step_current: int,
        const_vel_z_sim: bool,
        const_vel_no_sim: bool,
        w_road_edge: float,
        use_wosac_col: bool,
    ) -> None:
        super().__init__()
        self.step_gt = step_gt
        self.step_current = step_current
        self.const_vel_z_sim = const_vel_z_sim
        self.const_vel_no_sim = const_vel_no_sim
        self.n_joint_future = 32  # from WOSAC challenge
        self.w_road_edge = w_road_edge
        self.use_wosac_col = use_wosac_col

    def _filter_futures(self, buffer: RolloutBuffer, ag_role: Tensor) -> Tensor:
        """
        Args:
            buffer.pred_pose: [n_sc, n_K, n_ag, n_step, 3] (x,y,yaw)
            buffer.violation["*"]: [n_sc, n_K, n_ag, n_step]
            # not used
            buffer.violation["outside_map"]
            buffer.violation["passive"]
            buffer.violation["goal_reached"]
            buffer.violation["dest_reached"]
        Returns:
            trajs: [n_sc, self.n_joint_future, n_ag, n_step, 3] (x,y,yaw)
        """
        trajs = buffer.pred_pose[:, :, :, buffer.step_future_start :]

        if trajs.shape[1] > self.n_joint_future:

            ag_role = (ag_role.any(-1) * 1.0).unsqueeze(1)  # [n_sc, 1, n_ag]
            # [n_sc, n_K, n_ag]
            k_col = "collided_wosac" if self.use_wosac_col else "collided"
            collided = buffer.violation[k_col][..., buffer.step_future_start :].any(-1)
            run_road_edge = buffer.violation["run_road_edge"][..., buffer.step_future_start :].any(-1)

            collided = (collided * ag_role).sum(-1)
            run_road_edge = (run_road_edge * ag_role).sum(-1)

            violation = collided + run_road_edge * self.w_road_edge

            # select futures with minimum violation, idx: [n_sc, self.n_joint_future]
            _, idx = torch.topk(violation, self.n_joint_future, dim=-1, largest=False, sorted=False)

            trajs = trajs[torch.arange(trajs.shape[0]).unsqueeze(1), idx]

        return trajs

    def forward(self, batch: Dict[str, Tensor], buffer: RolloutBuffer) -> Dict[str, Tensor]:
        trajs = self._filter_futures(buffer, batch["ref/ag_role"])

        scenario_center = batch["scenario_center"].unsqueeze(1)  # [n_sc, 1, 2]
        scenario_rot = torch_rad2rot(batch["scenario_yaw"])  # [n_sc, 2, 2]

        pos_sim, yaw_sim = trajs[..., :2], trajs[..., 2:3]  # [n_sc, n_joint_future, n_ag, n_step_future, 2/1]
        pos_sim = torch_pos2global(pos_sim.flatten(1, 3), scenario_center, scenario_rot).view(pos_sim.shape)
        yaw_sim = torch_rad2global(yaw_sim.flatten(1, 4), batch["scenario_yaw"]).view(yaw_sim.shape)

        pos_no_sim = batch["history/agent_no_sim/pos"][..., :2]  # [n_sc, n_ag_no_sim, n_step_history, 3]
        yaw_no_sim = batch["history/agent_no_sim/yaw_bbox"]  # [n_sc, n_ag_no_sim, n_step_history, 1]
        pos_no_sim = torch_pos2global(pos_no_sim.flatten(1, 2), scenario_center, scenario_rot).view(pos_no_sim.shape)
        yaw_no_sim = torch_rad2global(yaw_no_sim.flatten(1, 3), batch["scenario_yaw"]).view(yaw_no_sim.shape)

        scenario_id = []
        for str_id in batch["scenario_id"]:
            int_id = [-1] * 16  # max_len of scenario_id string is 16
            for i, c in enumerate(str_id):
                int_id[i] = ord(c)
            scenario_id.append(torch.tensor(int_id, dtype=torch.int32, device=trajs.device).unsqueeze(0))

        wosac_data = {
            "scenario_id": torch.cat(scenario_id, dim=0),  # [n_sc, 16]
            "valid_sim": batch["history/agent/valid"],
            "pos_sim": pos_sim,  # [n_sc, n_joint_future, n_ag, n_step_future, 2]
            "z_sim": batch["history/agent/pos"][..., 2:3],
            "yaw_sim": yaw_sim,  # [n_sc, n_joint_future, n_ag, n_step_future, 1]
            "valid_no_sim": batch["history/agent_no_sim/valid"],
            "object_id_sim": batch["history/agent/object_id"],
            "pos_no_sim": pos_no_sim,  # [n_sc, n_ag_no_sim, n_step_history, 2]
            "z_no_sim": batch["history/agent_no_sim/pos"][..., 2:3],
            "yaw_no_sim": yaw_no_sim,  # [n_sc, n_ag_no_sim, n_step_history, 1]
            "object_id_no_sim": batch["history/agent_no_sim/object_id"],
        }
        return wosac_data

    def get_scenario_rollouts(self, wosac_data: Dict[str, Tensor]) -> List[sim_agents_submission_pb2.ScenarioRollouts]:
        for k in wosac_data.keys():
            if type(wosac_data[k]) is list:  # single gpu fix
                wosac_data[k] = wosac_data[k][0]
            wosac_data[k] = wosac_data[k].cpu().numpy()

        n_sc, n_joint_future = wosac_data["pos_sim"].shape[0], wosac_data["pos_sim"].shape[1]
        scenario_rollouts = []
        for i in range(n_sc):
            traj_no_sim = self._get_no_sim_joint_scene(
                valid=wosac_data["valid_no_sim"][i],
                pos=wosac_data["pos_no_sim"][i],
                z=wosac_data["z_no_sim"][i],
                yaw=wosac_data["yaw_no_sim"][i],
                object_id=wosac_data["object_id_no_sim"][i],
            )

            joint_scenes = []
            for i_rollout in range(n_joint_future):
                traj_sim = self._get_sim_joint_scene(
                    valid=wosac_data["valid_sim"][i],  # [n_ag, n_step_history]
                    pos=wosac_data["pos_sim"][i, i_rollout, :],  # [n_ag, n_step_future, 2]
                    z=wosac_data["z_sim"][i],  # [n_ag, n_step_history, 1]
                    yaw=wosac_data["yaw_sim"][i, i_rollout, :],  # [n_ag, n_step_future, 1]
                    object_id=wosac_data["object_id_sim"][i],  # [n_ag]
                )

                traj_all = sim_agents_submission_pb2.JointScene(simulated_trajectories=traj_sim + traj_no_sim)
                # submission_specs.validate_joint_scene(traj_all, scenario)
                joint_scenes.append(traj_all)

            scenario_id = "".join([chr(x) for x in wosac_data["scenario_id"][i] if x > 0])

            scenario_rollouts.append(
                sim_agents_submission_pb2.ScenarioRollouts(joint_scenes=joint_scenes, scenario_id=scenario_id)
            )

        return scenario_rollouts

    def _get_sim_joint_scene(
        self, valid: np.ndarray, pos: np.ndarray, yaw: np.ndarray, z: np.ndarray, object_id: np.ndarray
    ) -> List[sim_agents_submission_pb2.SimulatedTrajectory]:
        """
        Args: numpy
            valid: [n_ag, n_step_history]
            pos: [n_ag, n_step_future, 2]
            yaw: [n_ag, n_step_future, 1]
            z: [n_ag, n_step_history, 1]
            object_id: [n_ag]
        """
        t_step = np.arange(self.step_gt - self.step_current) + 1
        simulated_trajectories = []
        for i in np.where(valid[:, self.step_current])[0]:
            if self.const_vel_z_sim and (valid[i, self.step_current] and valid[i, self.step_current - 1]):
                v_z = z[i, self.step_current, 0] - z[i, self.step_current - 1, 0]
            else:
                v_z = 0.0

            simulated_trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pos[i, :, 0],
                    center_y=pos[i, :, 1],
                    center_z=z[i, self.step_current, 0] + v_z * t_step,
                    heading=yaw[i, :, 0],
                    object_id=object_id[i],
                )
            )
        return simulated_trajectories

    def _get_no_sim_joint_scene(
        self, valid: np.ndarray, pos: np.ndarray, yaw: np.ndarray, z: np.ndarray, object_id: np.ndarray
    ) -> List[sim_agents_submission_pb2.SimulatedTrajectory]:
        """
        Args: numpy
            valid: [n_agent_no_sim, n_step_history]
            pos: [n_agent_no_sim, n_step_history, 2]
            yaw: [n_agent_no_sim, n_step_history, 1]
            z: [n_agent_no_sim, n_step_history, 1]
            object_id: [n_agent_no_sim]
        """
        t_step = np.arange(self.step_gt - self.step_current) + 1
        simulated_trajectories = []
        for i in np.where(valid[:, self.step_current])[0]:
            if self.const_vel_no_sim and (valid[i, self.step_current] and valid[i, self.step_current - 1]):
                v_x = pos[i, self.step_current, 0] - pos[i, self.step_current - 1, 0]
                v_y = pos[i, self.step_current, 1] - pos[i, self.step_current - 1, 1]
                v_z = z[i, self.step_current, 0] - z[i, self.step_current - 1, 0]
            else:
                v_x, v_y, v_z = 0.0, 0.0, 0.0

            simulated_trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pos[i, self.step_current, 0] + v_x * t_step,
                    center_y=pos[i, self.step_current, 1] + v_y * t_step,
                    center_z=z[i, self.step_current, 0] + v_z * t_step,
                    heading=np.tile(yaw[i, self.step_current, 0], len(t_step)),
                    object_id=object_id[i],
                )
            )
        return simulated_trajectories
