# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Tuple, Optional
from torch import Tensor
import numpy as np
import torch
from utils.transform_utils import cast_rad
from .wosac_collision import get_ag_bbox, check_collided_wosac


class TrafficRuleChecker:
    def __init__(
        self,
        mp_boundary: Tensor,
        mp_valid: Tensor,
        mp_type: Tensor,
        mp_pos: Tensor,
        mp_dir: Tensor,
        ag_type: Tensor,
        ag_size: Tensor,
        ag_goal: Optional[Tensor],
        ag_dest: Optional[Tensor],
        tl_valid: Tensor,  # [n_sc, n_tl]
        tl_pose: Tensor,  #  [n_sc, n_tl, 3]
        disable_check: bool,
        collision_size_scale: float = 1.1,
    ) -> None:
        self.ag_size = ag_size[..., :2] * collision_size_scale
        self.mp_boundary = mp_boundary
        self.disable_check = disable_check
        mp_pos, mp_dir = mp_pos[..., :2], mp_dir[..., :2]

        self.tl_valid, self.tl_pose = tl_valid, tl_pose

        mask = ag_type[:, :, 0]
        n_sc, n_ag = mask.shape
        self.outside_map = torch.zeros_like(mask)
        self.collided = torch.zeros_like(mask)
        self.collided_wosac = torch.zeros_like(mask)
        self.run_red_light = torch.zeros_like(mask)
        self.goal_reached = torch.zeros_like(mask)
        self.dest_reached = torch.zeros_like(mask)
        self.run_road_edge = torch.zeros_like(mask)
        self.passive = torch.zeros_like(mask)
        self.passive_counter = torch.zeros_like(mask, dtype=torch.float32)

        # for self._check_collided
        # [n_sc, n_ag, n_ag]: no self collision
        self.ag_ego_mask = torch.eye(n_ag, dtype=torch.bool, device=mask.device)[None, :, :].expand(n_sc, -1, -1)
        ped_cyc_mask = ag_type[:, :, 1]
        collision_ped_cyc_mask = ped_cyc_mask.unsqueeze(1) & ped_cyc_mask.unsqueeze(2)
        self.collision_invalid_mask = self.ag_ego_mask | collision_ped_cyc_mask

        # for self._check_run_road_edge
        self.road_edge, self.road_edge_valid = self._get_road_edge(mp_valid, mp_type, mp_pos, mp_dir)

        # for self._check_run_red_light
        self.run_red_light_agent_length = ag_size[:, :, [0]] * 0.5 * 0.6  # [n_sc, n_ag, 1]
        self.run_red_light_agent_width = ag_size[:, :, [1]] * 0.5 * 1.8  # [n_sc, n_ag, 1]
        self.veh_mask = ag_type[:, :, 0]  # [n_sc, n_ag]

        # for self._check_passive
        self.lane_center, self.lane_center_valid = self._get_lane_center(mp_valid, mp_type, mp_pos)

        # for self._check_goal_reached
        self.ag_goal = ag_goal
        self.goal_thresh_pos = ag_size[:, :, 0] * 8  # [n_sc, n_ag]
        self.goal_thresh_rot = np.deg2rad(15)

        # for self._check_dest_reached
        if ag_dest is None:
            self.dest = None
        else:
            self.mp_valid, self.mp_type = mp_valid, mp_type
            self.mp_pos, self.mp_dir = mp_pos[..., :2], mp_dir[..., :2]
            self.dest = self._get_dest(
                ag_dest=ag_dest,
                mp_valid=self.mp_valid,
                mp_type=self.mp_type,
                mp_pos=self.mp_pos,
                mp_dir=self.mp_dir,
                ag_size=self.ag_size,
            )

    @staticmethod
    def _get_dest(
        mp_valid: Tensor, ag_dest: Tensor, mp_type: Tensor, mp_pos: Tensor, mp_dir: Tensor, ag_size: Tensor,
    ) -> Dict[str, Tensor]:
        batch_idx = torch.arange(mp_valid.shape[0]).unsqueeze(1)  # [n_sc, 1]

        dest_type = mp_type[batch_idx, ag_dest]  # [n_sc, n_ag, n_type]: onehot LANE<=3, ROAD_EDGE_BOUNDARY = 4
        dest_dir = mp_dir[batch_idx, ag_dest]
        dest_dir = dest_dir / torch.norm(dest_dir, dim=-1, keepdim=True)

        # dest_thresh_pos: [n_sc, n_ag], thresh_lane=50, thresh_edge=10
        dest_thresh_pos = torch.ones_like(ag_size[:, :, 0]) * 50
        dest_thresh_pos = dest_thresh_pos * (1 - dest_type[:, :, 4] * 0.8)
        return {
            "dest_invalid": ~(mp_valid[batch_idx, ag_dest]),  # [n_sc, n_ag, 20]
            "dest_type": dest_type,  # [n_sc, n_ag, n_type]
            "dest_pos": mp_pos[batch_idx, ag_dest],  # [n_sc, n_ag, 20, 2]
            "dest_dir": dest_dir,  # [n_sc, n_ag, 20, 2]
            "dest_thresh_rot": np.deg2rad(30),  # only valid for lane dests
            "dest_thresh_pos": dest_thresh_pos,  # [n_sc, n_ag]
        }

    @staticmethod
    def _check_outside_map(
        valid: Tensor,  # [n_sc, n_ag] bool
        pose: Tensor,  # [n_sc, n_ag, 3] (x,y,yaw)
        mp_boundary: Tensor,  # [n_sc, 4], xmin,xmax,ymin,ymax
    ) -> Tensor:  # outside_map_this_step: [n_sc, n_ag], bool
        x, y = pose[:, :, 0], pose[:, :, 1]  # [n_sc, n_ag]
        # [n_sc, 1]
        xmin, xmax, ymin, ymax = mp_boundary[:, [0]], mp_boundary[:, [1]], mp_boundary[:, [2]], mp_boundary[:, [3]]
        outside_map_this_step = ((x > xmax) | (x < xmin) | (y > ymax) | (y < ymin)) & valid
        return outside_map_this_step

    @staticmethod
    def _check_collided(
        valid: Tensor,  # [n_sc, n_ag] bool
        bbox: Tensor,  # [n_sc, n_ag, 4, 2], 4 corners, (x,y)
        collision_invalid_mask: Tensor,  # [n_sc, n_ag, n_ag]: no self collision
    ) -> Tensor:  # collided_this_step: [n_sc, n_ag], bool
        bbox_next = bbox.roll(-1, dims=2)

        bbox_line = torch.cat(  # ax+by+c=0
            [
                bbox_next[..., [1]] - bbox[..., [1]],  # a
                bbox[..., [0]] - bbox_next[..., [0]],  # b
                bbox_next[..., [0]] * bbox[..., [1]] - bbox_next[..., [1]] * bbox[..., [0]],
            ],  # c
            axis=-1,
        )  # [n_sc, n_ag, 4, 3]
        bbox_point = torch.cat([bbox, torch.ones_like(bbox[..., [0]])], axis=-1)

        n_ag = bbox.shape[1]
        bbox_line = bbox_line[:, :, None, :, None, :].expand(-1, -1, n_ag, -1, 4, -1)  # [n_sc, n_ag, n_ag, 4, 4, 3]
        bbox_point = bbox_point[:, None, :, None, :, :].expand(-1, n_ag, -1, 4, -1, -1)

        is_outside = torch.sum(bbox_line * bbox_point, axis=-1) > 0  # [n_sc, n_ag, n_ag, 4, 4]

        no_collision = torch.any(torch.all(is_outside, axis=-1), axis=-1)  # [n_sc, n_ag, n_ag]
        no_collision = no_collision | no_collision.transpose(1, 2)

        # [n_sc, n_ag, n_ag]: no collision for invalid agent
        invalid_mask = ~(valid[:, :, None] & valid[:, None, :])
        no_collision = no_collision | collision_invalid_mask | invalid_mask
        collided_this_step = ~(no_collision.all(-1))
        return collided_this_step

    @staticmethod
    def _check_run_road_edge(
        valid: Tensor,  # [n_sc, n_ag] bool
        bbox: Tensor,  # [n_sc, n_ag, 4, 2], 4 corners, (x,y)
        veh_mask: Tensor,  # [n_sc, n_ag] bool
        road_edge: Tensor,  # [n_sc, n_mp*20, 2, 2], (start/end), (x,y)
        road_edge_valid: Tensor,  # [n_sc, n_mp*20], bool
    ) -> Tensor:  # run_road_edge_this_step: [n_sc, n_ag], bool
        bbox_next = bbox.roll(-1, dims=2)  # [n_sc, n_ag, 4, 2]
        bbox_line = torch.stack([bbox, bbox_next], dim=-2).unsqueeze(2)  # [n_sc, n_ag, 1, 4, 2, 2]
        road_edge_line = road_edge[:, None, :, None, :, :]  # [n_sc, n_mp*20, 2, 2] -> [n_sc, 1, n_mp*20, 1, 2, 2]

        # [n_sc, n_ag, n_mp*20, 4, 2]
        A, B = bbox_line[:, :, :, :, 0], bbox_line[:, :, :, :, 1]
        C, D = road_edge_line[:, :, :, :, 0], road_edge_line[:, :, :, :, 1]

        # [n_sc, n_ag, n_mp*20, 4]
        run_road_edge_this_step = (ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D))
        # [n_sc, n_ag, n_mp*20]
        run_road_edge_this_step = run_road_edge_this_step.any(-1) & road_edge_valid.unsqueeze(1)
        # [n_sc, n_ag]
        run_road_edge_this_step = run_road_edge_this_step.any(-1) & valid & veh_mask
        return run_road_edge_this_step

    @staticmethod
    def _check_run_red_light(
        valid: Tensor,  # [n_sc, n_ag], bool
        pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        tl_valid: Tensor,  # [n_sc, n_tl]
        tl_pose: Tensor,  #  [n_sc, n_tl, 3], (x,y,yaw)
        tl_state: Tensor,  # [n_sc, n_tl, tl_state_dim=5], bool one_hot
        run_red_light_agent_length: Tensor,  # [n_sc, n_ag, 1]
        run_red_light_agent_width: Tensor,  # [n_sc, n_ag, 1]
        veh_mask: Tensor,  # [n_sc, n_ag], bool
    ) -> Tensor:  # run_red_light_this_step: [n_sc, n_ag], bool
        """
        tl_state:
            LANE_STATE_UNKNOWN = 0;
            LANE_STATE_STOP = 1;
            LANE_STATE_CAUTION = 2;
            LANE_STATE_GO = 3;
            LANE_STATE_FLASHING = 4;
        """
        heading_cos, heading_sin = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])  # [n_sc, n_ag]
        # [n_sc, n_ag, 1, 2]
        heading_f = torch.stack([heading_cos, heading_sin], axis=-1).unsqueeze(2)
        heading_r = torch.stack([heading_sin, -heading_cos], axis=-1).unsqueeze(2)

        # [n_sc, n_ag, 1, 2]
        xy_0 = pose[..., :2].unsqueeze(2)
        xy_1 = xy_0 + 0.1 * motion[..., [0]].unsqueeze(2) * heading_f

        tl_pose = tl_pose[:, None, :, :2]  # [n_sc, n_tl, 3] -> [n_sc, 1, n_tl, 2]
        # [n_sc, n_ag, n_tl]
        inside_0 = torch.logical_and(
            torch.abs(torch.sum((tl_pose - xy_0) * heading_f, dim=-1)) < run_red_light_agent_length,
            torch.abs(torch.sum((tl_pose - xy_0) * heading_r, dim=-1)) < run_red_light_agent_width,
        )
        inside_1 = torch.logical_and(
            torch.abs(torch.sum((tl_pose - xy_1) * heading_f, dim=-1)) < run_red_light_agent_length,
            torch.abs(torch.sum((tl_pose - xy_1) * heading_r, dim=-1)) < run_red_light_agent_width,
        )
        mask_valid_agent = (valid & veh_mask).unsqueeze(2)  # [n_sc, n_ag, 1]
        mask_valid_tl = (tl_valid & tl_state[:, :, 1]).unsqueeze(1)  # [n_sc, 1, n_tl]
        run_red_light_this_step = inside_0 & (~inside_1) & mask_valid_agent & mask_valid_tl  # [n_sc, n_ag, n_tl]
        run_red_light_this_step = run_red_light_this_step.any(-1)  # [n_sc, n_ag]
        return run_red_light_this_step

    @staticmethod
    def _check_passive(
        valid: Tensor,  # [n_sc, n_ag], bool
        pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        tl_valid: Tensor,  # [n_sc, n_tl]
        tl_pose: Tensor,  #  [n_sc, n_tl, 3], (x,y,yaw)
        tl_state: Tensor,  # [n_sc, n_tl, tl_state_dim=5], bool one_hot
        lane_center: Tensor,  # [n_sc, n_mp*20, 2]
        lane_center_valid: Tensor,  # [n_sc, n_mp*20]
        veh_mask: Tensor,  # [n_sc, n_ag], bool
        ag_ego_mask: Tensor,  # [n_sc, n_ag, n_ag] bool, eye
        passive_counter: Tensor,  # [n_sc, n_ag], float32
    ) -> Tuple[Tensor, Tensor]:  # passive_this_step: [n_sc, n_ag] bool, passive_counter: [n_sc, n_ag] float32
        """
        tl_state:
            LANE_STATE_UNKNOWN = 0; CHECK
            LANE_STATE_STOP = 1; CHECK
            LANE_STATE_CAUTION = 2; CHECK
            LANE_STATE_GO = 3;
            LANE_STATE_FLASHING = 4; CHECK
        """
        # [n_sc, n_ag, 1, 2] - [n_sc, 1, n_mp*20, 2] = [n_sc, n_ag, n_mp*20, 2]
        close_to_lane = torch.norm(pose[:, :, :2].unsqueeze(2) - lane_center.unsqueeze(1), dim=-1)
        close_to_lane = close_to_lane < 2  # meter, [n_sc, n_ag, n_mp*20] bool
        # [n_sc, n_ag]: [n_sc, n_ag, n_mp*20] & [n_sc, 1, n_mp*20]
        close_to_lane = (close_to_lane & lane_center_valid.unsqueeze(1)).any(-1)

        low_speed = motion[:, :, 0] < 5  # meter/second, [n_sc, n_ag]

        # [n_sc, n_ag, 1, 2]
        heading_f = torch.stack([torch.cos(pose[..., 2]), torch.sin(pose[..., 2])], axis=-1).unsqueeze(2)

        # check red (flashing,yellow) light ahead
        mask_valid_tl = (tl_valid & tl_state[:, :, [0, 1, 2, 4]].any(-1)).unsqueeze(1)  # [n_sc, 1, n_tl]
        tl_vec = tl_pose[:, None, :, :2] - pose[:, :, :2].unsqueeze(2)  # [n_sc, n_ag, n_tl, 2]
        tl_vec_norm = torch.norm(tl_vec, dim=-1)
        tl_is_close = tl_vec_norm < 10  # meter, [n_sc, n_ag, n_tl]
        tl_is_ahead = ((heading_f * tl_vec).sum(-1) / tl_vec_norm) > 0.95  # np.cos(np.deg2rad(18)) = 0.95
        red_tl_ahead = (tl_is_close & tl_is_ahead & mask_valid_tl).any(-1)  # [n_sc, n_ag]

        # check other agents
        agent_vec = pose[:, :, :2].unsqueeze(1) - pose[:, :, :2].unsqueeze(2)  # [n_sc, n_ag, n_ag, 2]
        agent_vec_norm = torch.norm(agent_vec, dim=-1)  # [n_sc, n_ag, n_ag]
        ag_is_close = agent_vec_norm < 10  # meter, [n_sc, n_ag, n_ag]
        ag_is_ahead = ((heading_f * agent_vec).sum(-1) / agent_vec_norm) > 0.95
        # [n_sc, n_ag]
        agent_ahead = (ag_is_close & ag_is_ahead & valid.unsqueeze(1) & valid.unsqueeze(2) & (~ag_ego_mask)).any(-1)

        passive_this_step = valid & veh_mask & close_to_lane & low_speed & (~red_tl_ahead) & (~agent_ahead)

        # accumulate, set to zero if not passive
        passive_counter = (passive_counter + passive_this_step) * passive_this_step
        passive_this_step = passive_counter > 20
        return passive_this_step, passive_counter

    @staticmethod
    def _check_goal_reached(
        valid: Tensor,  # [n_sc, n_ag], bool
        pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        goal: Tensor,  # [n_sc, n_ag, 4], x,y,yaw,v
        goal_reached: Tensor,  # [n_sc, n_ag], bool
        goal_thresh_pos: Tensor,  # [n_sc, n_ag] agent_length * 8
        goal_thresh_rot: float,  # 15 rad
    ) -> Tensor:  # goal_reached_this_step: [n_sc, n_ag], bool
        pos_reached = torch.norm(pose[..., :2] - goal[..., :2], dim=-1) < goal_thresh_pos  # [n_sc, n_ag]
        rot_reached = torch.abs(cast_rad(pose[..., 2] - goal[..., 2])) < goal_thresh_rot  # [n_sc, n_ag]
        goal_reached_this_step = pos_reached & rot_reached & valid & (~goal_reached)
        return goal_reached_this_step

    @staticmethod
    def _check_dest_reached(
        valid: Tensor,  # [n_sc, n_ag], bool
        pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        dest_invalid: Tensor,  # [n_sc, n_ag, 20]
        dest_type: Tensor,  # [n_sc, n_ag, n_mp_type] one_hot bool, LANE<=3, ROAD_EDGE = 4
        dest_pos: Tensor,  # [n_sc, n_ag, 20, 2]
        dest_dir: Tensor,  # [n_sc, n_ag, 20, 2], unit_vec
        dest_reached: Tensor,  # [n_sc, n_ag], bool
        dest_thresh_pos: Tensor,  # [n_sc, n_ag]
        dest_thresh_rot: float,  # in rad
    ) -> Tensor:  # dest_reached_this_step: [n_sc, n_ag], bool
        # [n_sc, n_ag, 20]
        dist_to_dest = torch.norm(pose[..., :2].unsqueeze(2) - dest_pos, dim=-1).masked_fill(dest_invalid, float("inf"))

        pos_reached = (dist_to_dest < dest_thresh_pos.unsqueeze(-1)).any(-1)  # [n_sc, n_ag]
        heading_f = torch.stack([torch.cos(pose[..., 2]), torch.sin(pose[..., 2])], axis=-1)  # [n_sc, n_ag, 2]

        # [n_sc, n_ag, 1, 2] * [n_sc, n_ag, 20, 2]
        rot_diff_to_dest = (heading_f.unsqueeze(2) * dest_dir).sum(-1).masked_fill(dest_invalid, 0)  # [n_sc, n_ag, 20]
        # [n_sc, n_ag]
        rot_reached = (rot_diff_to_dest > np.cos(dest_thresh_rot)).any(-1)

        # [n_sc, n_ag]: one_hot bool, LANE<=3, TYPE_ROAD_EDGE_BOUNDARY = 4
        mask_lane = dest_type[:, :, :4].any(-1)
        mask_edge = dest_type[:, :, 4]
        dest_reached_this_step = (
            (~dest_reached) & valid & ((mask_lane & pos_reached & rot_reached) | (mask_edge & pos_reached))
        )
        return dest_reached_this_step

    @torch.no_grad()
    def update_navi(
        self,
        navi_mode: str,  # "goal" or "dest"
        mask_new_navi: Tensor,  #  [n_sc, n_ag], bool
        navi: Tensor,  # goal [n_sc, n_ag, 4], dest [n_sc, n_ag]
    ) -> None:
        if navi_mode == "dest":
            self.dest = self._get_dest(
                ag_dest=navi,
                mp_valid=self.mp_valid,
                mp_type=self.mp_type,
                mp_pos=self.mp_pos,
                mp_dir=self.mp_dir,
                ag_size=self.ag_size,
            )
            self.dest_reached = self.dest_reached & (~mask_new_navi)
        elif navi_mode == "goal":
            self.ag_goal = navi
            self.goal_reached = self.goal_reached & (~mask_new_navi)

    @torch.no_grad()
    def check(
        self,
        valid: Tensor,  # [n_sc, n_ag], bool
        pose: Tensor,  # [n_sc, n_ag, 3], (x,y,yaw)
        motion: Tensor,  # [n_sc, n_ag, 3], (spd,acc,yaw_rate)
        tl_state: Tensor,  # [n_sc, n_tl, tl_state_dim], bool one_hot
    ) -> Dict[str, Tensor]:  # violations for this current step: Dict {str -> Tensor [n_sc, n_ag]}

        bbox = get_ag_bbox(pose, self.ag_size)

        outside_map_this_step = self._check_outside_map(valid, pose, self.mp_boundary)
        self.outside_map = self.outside_map | outside_map_this_step

        if self.disable_check:
            collided_this_step = self.collided
        else:
            collided_this_step = self._check_collided(valid, bbox, self.collision_invalid_mask)
            self.collided = self.collided | collided_this_step

        if self.disable_check:
            collided_wosac_this_step = self.collided_wosac
        else:
            collided_wosac_this_step = check_collided_wosac(pose, self.ag_size, valid)
            self.collided_wosac = self.collided_wosac | collided_wosac_this_step

        if self.disable_check:
            run_road_edge_this_step = self.run_road_edge
        else:
            run_road_edge_this_step = self._check_run_road_edge(
                valid, bbox, self.veh_mask, self.road_edge, self.road_edge_valid
            )
            self.run_road_edge = self.run_road_edge | run_road_edge_this_step

        # step can be larger than ground truth tl_step
        if self.disable_check:
            run_red_light_this_step = self.run_red_light
        else:
            run_red_light_this_step = self._check_run_red_light(
                valid=valid,
                pose=pose,
                motion=motion,
                tl_valid=self.tl_valid,
                tl_pose=self.tl_pose,
                tl_state=tl_state,
                run_red_light_agent_length=self.run_red_light_agent_length,
                run_red_light_agent_width=self.run_red_light_agent_width,
                veh_mask=self.veh_mask,
            )
            self.run_red_light = self.run_red_light | run_red_light_this_step

        if self.disable_check:
            passive_this_step = self.passive
        else:
            passive_this_step, self.passive_counter = self._check_passive(
                valid=valid,
                pose=pose,
                motion=motion,
                tl_valid=self.tl_valid,
                tl_pose=self.tl_pose,
                tl_state=tl_state,
                lane_center=self.lane_center,
                lane_center_valid=self.lane_center_valid,
                veh_mask=self.veh_mask,
                ag_ego_mask=self.ag_ego_mask,
                passive_counter=self.passive_counter,
            )
            self.passive = self.passive | passive_this_step

        if self.ag_goal is None:
            goal_reached_this_step = torch.zeros_like(self.goal_reached)
        else:
            goal_reached_this_step = self._check_goal_reached(
                valid=valid,
                pose=pose,
                goal=self.ag_goal,
                goal_reached=self.goal_reached,
                goal_thresh_pos=self.goal_thresh_pos,
                goal_thresh_rot=self.goal_thresh_rot,
            )
        self.goal_reached = self.goal_reached | goal_reached_this_step

        if self.dest is None:
            dest_reached_this_step = torch.zeros_like(self.dest_reached)
        else:
            dest_reached_this_step = self._check_dest_reached(
                valid=valid, pose=pose, dest_reached=self.dest_reached, **self.dest
            )
        self.dest_reached = self.dest_reached | dest_reached_this_step

        # [n_sc, n_ag], bool
        violations = {
            "outside_map": self.outside_map,
            "outside_map_this_step": outside_map_this_step,
            "collided": self.collided,  # no collision ped2ped
            "collided_this_step": collided_this_step,
            "collided_wosac": self.collided_wosac,  # no collision ped2ped
            "collided_wosac_this_step": collided_wosac_this_step,
            "run_road_edge": self.run_road_edge,  # only for vehicles
            "run_road_edge_this_step": run_road_edge_this_step,
            "run_red_light": self.run_red_light,  # only for vehicles
            "run_red_light_this_step": run_red_light_this_step,
            "passive": self.passive,  # only for vehicles
            "passive_this_step": passive_this_step,
            "goal_reached": self.goal_reached,
            "goal_reached_this_step": goal_reached_this_step,
            "dest_reached": self.dest_reached,
            "dest_reached_this_step": dest_reached_this_step,
        }
        return violations

    @staticmethod
    def _get_road_edge(
        mp_valid: Tensor,  # [n_sc, n_mp, 20]
        mp_type: Tensor,  # [n_sc, n_mp, 11] one_hot bool
        mp_pos: Tensor,  # [n_sc, n_mp, 20, 2]
        mp_dir: Tensor,  # [n_sc, n_mp, 20, 2]
    ) -> Tuple[Tensor, Tensor]:
        """
            mp_type:
                # FREEWAY = 0
                # SURFACE_STREET = 1
                # STOP_SIGN = 2
                # BIKE_LANE = 3
                # TYPE_ROAD_EDGE_BOUNDARY = 4
                # TYPE_ROAD_EDGE_MEDIAN = 5
                # SOLID_SINGLE = 6
                # SOLID_DOUBLE = 7
                # PASSING_DOUBLE_YELLOW = 8
                # SPEED_BUMP = 9
                # CROSSWALK = 10
            
        Returns:
            road_edge: [n_sc, n_mp*20, 2, 2], (start/end), (x,y)
            road_edge_valid: [n_sc, n_mp*20], bool
        """
        road_edge_valid = mp_valid & mp_type[:, :, [4, 5, 7]].any(dim=-1, keepdim=True)  # [n_sc, n_mp, 20]
        road_edge = torch.stack([mp_pos, mp_pos + mp_dir], dim=-2)  # [n_sc, n_mp, 20, 2] -> [n_sc, n_mp, 20, 2, 2]
        return road_edge.flatten(1, 2), road_edge_valid.flatten(1, 2)

    @staticmethod
    def _get_lane_center(mp_valid: Tensor, mp_type: Tensor, mp_pos: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            mp_valid: [n_sc, n_mp, 20]
            mp_type: [n_sc, n_mp]
            mp_pos: [n_sc, n_mp, 20, 2]

        Returns:
            lane_center: [n_sc, n_mp*20, 2]
            lane_center_valid: [n_sc, n_mp*20]
        """
        lane_center_valid = mp_type[:, :, :3].any(dim=-1, keepdim=True)  # [n_sc, n_mp, 1]
        lane_center_valid = mp_valid & lane_center_valid  # [n_sc, n_mp, 20]
        return mp_pos.flatten(1, 2), lane_center_valid.flatten(1, 2)


def ccw(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
