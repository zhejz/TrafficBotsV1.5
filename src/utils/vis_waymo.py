# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .video_recorder import ImageEncoder

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_VIOLET = (170, 0, 255)

COLOR_BUTTER_0 = (252, 233, 79)
COLOR_BUTTER_1 = (237, 212, 0)
COLOR_BUTTER_2 = (196, 160, 0)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_ORANGE_1 = (245, 121, 0)
COLOR_ORANGE_2 = (209, 92, 0)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHOCOLATE_1 = (193, 125, 17)
COLOR_CHOCOLATE_2 = (143, 89, 2)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_CHAMELEON_1 = (115, 210, 22)
COLOR_CHAMELEON_2 = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (52, 101, 164)
COLOR_SKY_BLUE_2 = (32, 74, 135)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_PLUM_1 = (117, 80, 123)
COLOR_PLUM_2 = (92, 53, 102)
COLOR_SCARLET_RED_0 = (239, 41, 41)
COLOR_SCARLET_RED_1 = (204, 0, 0)
COLOR_SCARLET_RED_2 = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (186, 189, 182)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_4 = (85, 87, 83)
COLOR_ALUMINIUM_4_5 = (66, 62, 64)
COLOR_ALUMINIUM_5 = (46, 52, 54)


class VisWaymo:
    def __init__(
        self,
        map_valid: np.ndarray,
        map_type: np.ndarray,
        map_pos: np.ndarray,
        map_boundary: np.ndarray,
        px_per_m: float = 10.0,
        video_size: int = 960,
    ) -> None:
        # centered around ego vehicle first step, x=0, y=0, theta=0
        self.px_per_m = px_per_m
        self.video_size = video_size
        self.px_agent2bottom = video_size // 2

        # waymo
        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_4_5, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE_2, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE_2, 6),  # BIKE_LANE = 3
            (COLOR_SKY_BLUE_2, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM_2, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER_0, 2),  # BROKEN = 6
            (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED_2, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON_2, 4),  # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0;
            COLOR_RED,  # STOP = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN,  # GO = 3;
            COLOR_VIOLET,  # FLASHING = 4;
        ]
        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON_2, COLOR_MAGENTA]

        self.agent_cmd_txt = [
            "STATIONARY",  # STATIONARY = 0;
            "STRAIGHT",  # STRAIGHT = 1;
            "STRAIGHT_LEFT",  # STRAIGHT_LEFT = 2;
            "STRAIGHT_RIGHT",  # STRAIGHT_RIGHT = 3;
            "LEFT_U_TURN",  # LEFT_U_TURN = 4;
            "LEFT_TURN",  # LEFT_TURN = 5;
            "RIGHT_U_TURN",  # RIGHT_U_TURN = 6;
            "RIGHT_TURN",  # RIGHT_TURN = 7;
        ]

        raster_map, self.top_left_px = self._register_map(map_boundary, self.px_per_m)
        self.raster_map = self._draw_map(raster_map, map_valid, map_type, map_pos)

    @staticmethod
    def _register_map(map_boundary: np.ndarray, px_per_m: float, edge_px: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            map_boundary: [4], xmin, xmax, ymin, ymax
            px_per_m: float

        Returns:
            raster_map: empty image
            top_left_px
        """
        # y axis is inverted in pixel coordinate
        xmin, xmax, ymax, ymin = (map_boundary * px_per_m).astype(np.int64)
        ymax *= -1
        ymin *= -1
        xmin -= edge_px
        ymin -= edge_px
        xmax += edge_px
        ymax += edge_px

        raster_map = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin], dtype=np.float32)
        return raster_map, top_left_px

    def _draw_map(
        self,
        raster_map: np.ndarray,
        map_valid: np.ndarray,
        map_type: np.ndarray,
        map_pos: np.ndarray,
        attn_weights_to_pl: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args: numpy arrays
            map_valid: [n_pl, 20],  # bool
            map_type: [n_pl, 11],  # bool one_hot
            map_pos: [n_pl, 20, 2],  # float32
            attn_weights_to_pl: [n_pl], sum up to 1

        Returns:
            raster_map
        """
        mask_valid = map_valid.any(axis=1)
        if attn_weights_to_pl is None:
            attn_weights_to_pl = np.zeros(map_valid.shape[0]) - 1

        for type_to_draw in range(len(self.lane_style)):
            for i in np.where((map_type[:, type_to_draw]) & mask_valid)[0]:
                color, thickness = self.lane_style[type_to_draw]
                if attn_weights_to_pl[i] > 0:
                    color = tuple(np.array(color) * attn_weights_to_pl[i])
                cv2.polylines(
                    raster_map,
                    [self._to_pixel(map_pos[i][map_valid[i]])],
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        # for i in range(mask_valid.shape[0]):
        #     if mask_valid[i]:
        #         cv2.arrowedLine(
        #             raster_map,
        #             self._to_pixel(map_pos[i, 0]),
        #             self._to_pixel(map_pos[i, 0] + (map_pos[i, 2] - map_pos[i, 0]) * 1),
        #             color=COLOR_CYAN,
        #             thickness=4,
        #             line_type=cv2.LINE_AA,
        #             tipLength=0.5,
        #         )
        return raster_map

    def save_prediction_videos(
        self,
        video_base_name: str,
        episode: Dict[str, np.ndarray],
        prediction: Optional[Dict[str, np.ndarray]],
        save_agent_view: bool = True,
    ) -> List[str]:
        """prediction["step_current"] <= prediction["step_gt"] <= prediction["step_end"]
        Args:
            episode: Dict for ground truth, for t = {0, ..., prediction["step_gt"]}, 11 steps for test, 91 for train/val
                "agent/valid": [n_ag, n_step/n_step_hist], bool
                "agent/pos": [n_ag, n_step/n_step_hist, 2], (x,y)
                "agent/yaw_bbox": [n_ag, n_step/n_step_hist, 1], [-pi, pi]
                "agent/role": [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
                "agent/size": [n_ag, 3], [length, width, height]
                "map/valid": [n_mp, n_mp_pl_node]
                "map/pos": [n_mp, n_mp_pl_node, 2], (x,y)
                "tl_lane/valid": [n_tl_lane, n_step/n_step_hist], bool
                "tl_lane/state": [n_tl_lane, n_step/n_step_hist, n_tl_state], bool one_hot
                "tl_lane/idx": [n_tl_lane], int, -1 means not valid
                "tl_stop/valid": [n_tl_stop, n_step/n_step_hist], bool
                "tl_stop/state": [n_tl_stop, n_step/n_step_hist, n_tl_state], bool, one_hot
                "tl_stop/pos": [n_tl_stop, 2], (x,y)
                "tl_stop/dir": [n_tl_stop, 2], (x,y)
                "agent/goal": not exist or [n_ag, 4], float32: (x,y,yaw,spd)
                "agent/dest": not exist or [n_ag], int64: index to map [0, n_mp]
            prediction: Dict, for t = {prediction["step_current"]+1, ..., prediction["step_end"]}, 80 steps
                "agent/valid": [n_ag, n_step_future]
                "agent/pos": [n_ag, n_step_future, 2]
                "agent/yaw_bbox": [n_ag, n_step_future, 1]
                "tl_lane/state": not exist or [n_tl_lane, n_step_future, n_tl_state], bool one_hot
                "tl_stop/state": not exist or [n_tl_stop, n_step_future, n_tl_state], bool, one_hot
                "agent/goal": not exist or [n_ag, n_step_future, 4], float32: (x,y,yaw,spd)
                "agent/dest": not exist or [n_ag, n_step_future], int64: index to map [0, n_mp]
                "ag_navi_valid": [n_ag, n_step_future]
        """
        buffer_video = {f"{video_base_name}-gt.mp4": [[], None]}  # [List[im], agent_id]
        if prediction is None:
            # step_current = step_end
            step_end = episode["agent/valid"].shape[1] - 1
            step_gt = step_end
        else:
            # step_current = prediction["step_current"]
            step_end = prediction["step_end"]
            step_gt = prediction["step_gt"]
            buffer_video[f"{video_base_name}-pd.mp4"] = [[], None]
            buffer_video[f"{video_base_name}-mix.mp4"] = [[], None]
            if save_agent_view:
                buffer_video[f"{video_base_name}-sdc.mp4"] = [[], np.where(episode["agent/role"][:, 0])[0][0]]
                for i in np.where(episode["agent/role"][:, 1])[0]:
                    buffer_video[f"{video_base_name}-int_{i}.mp4"] = [[], i]
                for i in np.where(episode["agent/role"][:, 2])[0]:
                    buffer_video[f"{video_base_name}-pre_{i}.mp4"] = [[], i]
                n_others_to_vis = 5
                idx_others = np.where(prediction["agent/valid"].any(1) & ~(episode["agent/role"].any(1)))[0]
                for i in idx_others[:n_others_to_vis]:
                    buffer_video[f"{video_base_name}-other_{i}.mp4"] = [[], i]

        for t in range(step_end + 1):
            step_image = self.raster_map.copy()
            t_pred = t - prediction["step_current"] - 1

            # ! get the tl_lane and tl_stop to draw
            if t_pred < 0 or prediction is None:  # draw ground truth when prediction is not available
                # [n_tl_lane], [n_tl_lane, n_tl_state]
                tl_lane_valid, tl_lane_state = episode["tl_lane/valid"][:, t], episode["tl_lane/state"][:, t]
                # [n_tl_stop], [n_tl_stop, n_tl_state]
                tl_stop_valid, tl_stop_state = episode["tl_stop/valid"][:, t], episode["tl_stop/state"][:, t]
            else:  # ground truth traffic light not available
                if "tl_lane/state" in prediction:  # model predicted tl_lane_state
                    tl_lane_valid = episode["tl_lane/valid"].any(-1)
                    tl_lane_state = prediction["tl_lane/state"][:, t_pred]
                else:  # no gt, no pred -> no vis
                    tl_lane_valid = np.zeros_like(episode["tl_lane/valid"][:, 0])
                if "tl_stop/state" in prediction:  # model predicted tl_stop_state
                    tl_stop_valid = episode["tl_stop/valid"].any(-1)
                    tl_stop_state = prediction["tl_stop/state"][:, t_pred]
                else:  # no gt, no pred -> no vis
                    tl_stop_valid = np.zeros_like(episode["tl_stop/valid"][:, 0])
            # ! draw loop for tl_lane
            for i in range(tl_lane_valid.shape[0]):
                if tl_lane_valid[i]:
                    lane_idx = episode["tl_lane/idx"][i]
                    tl_state = tl_lane_state[i].argmax()
                    pos = self._to_pixel(episode["map/pos"][lane_idx][episode["map/valid"][lane_idx]])
                    cv2.polylines(
                        step_image,
                        [pos],
                        isClosed=False,
                        color=self.tl_style[tl_state],
                        thickness=8,
                        lineType=cv2.LINE_AA,
                    )
                    if tl_state >= 1 and tl_state <= 3:
                        cv2.drawMarker(
                            step_image,
                            pos[-1],
                            color=self.tl_style[tl_state],
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=10,
                            thickness=6,
                        )
            # ! draw loop for tl_stop
            for i in range(tl_stop_valid.shape[0]):
                if tl_stop_valid[i]:
                    tl_state = tl_stop_state[i].argmax()
                    stop_point = self._to_pixel(episode["tl_stop/pos"][i])
                    stop_point_end = self._to_pixel(episode["tl_stop/pos"][i] + 5 * episode["tl_stop/dir"][i])
                    cv2.arrowedLine(
                        step_image,
                        stop_point,
                        stop_point_end,
                        color=self.tl_style[tl_state],
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.3,
                    )

            # ! draw gt agents
            step_image_gt, raster_blend_gt = step_image.copy(), np.zeros_like(step_image)
            if t <= step_gt:
                ag_valid = episode["agent/valid"][:, t]  # [n_ag]
                ag_pos = episode["agent/pos"][:, t]  # [n_ag, 2]
                ag_yaw_bbox = episode["agent/yaw_bbox"][:, t]  # [n_ag, 1]
                bbox_gt = self._to_pixel(self._get_agent_bbox(ag_valid, ag_pos, ag_yaw_bbox, episode["agent/size"]))
                heading_start = self._to_pixel(ag_pos[ag_valid])
                ag_yaw_bbox = ag_yaw_bbox[:, 0][ag_valid]
                heading_end = self._to_pixel(
                    ag_pos[ag_valid] + 1.5 * np.stack([np.cos(ag_yaw_bbox), np.sin(ag_yaw_bbox)], axis=-1)
                )
                agent_role = episode["agent/role"][ag_valid]
                for i in range(agent_role.shape[0]):
                    if not agent_role[i].any():
                        color = COLOR_ALUMINIUM_0
                    else:
                        color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                    cv2.fillConvexPoly(step_image_gt, bbox_gt[i], color=color)
                    cv2.fillConvexPoly(raster_blend_gt, bbox_gt[i], color=color)
                    cv2.arrowedLine(
                        step_image_gt,
                        heading_start[i],
                        heading_end[i],
                        color=COLOR_BLACK,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.6,
                    )
            buffer_video[f"{video_base_name}-gt.mp4"][0].append(step_image_gt)

            # ! draw prediction agents
            if prediction is not None:
                if t_pred >= 0:
                    step_image_pd = step_image.copy()
                    ag_valid = prediction["agent/valid"][:, t_pred]  # # [n_ag]
                    ag_pos = prediction["agent/pos"][:, t_pred]  # [n_ag, 2]
                    ag_yaw_bbox = prediction["agent/yaw_bbox"][:, t_pred]  # [n_ag, 1]
                    bbox_pred = self._to_pixel(
                        self._get_agent_bbox(ag_valid, ag_pos, ag_yaw_bbox, episode["agent/size"])
                    )
                    heading_start = self._to_pixel(ag_pos[ag_valid])
                    ag_yaw_bbox = ag_yaw_bbox[:, 0][ag_valid]
                    heading_end = self._to_pixel(
                        ag_pos[ag_valid] + 1.5 * np.stack([np.cos(ag_yaw_bbox), np.sin(ag_yaw_bbox)], axis=-1)
                    )
                    agent_role = episode["agent/role"][ag_valid]
                    for i in range(agent_role.shape[0]):
                        if not agent_role[i].any():
                            color = COLOR_ALUMINIUM_0
                        else:
                            color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                        cv2.fillConvexPoly(step_image_pd, bbox_pred[i], color=color)
                        cv2.arrowedLine(
                            step_image_pd,
                            heading_start[i],
                            heading_end[i],
                            color=COLOR_BLACK,
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.6,
                        )
                    step_image_mix = cv2.addWeighted(raster_blend_gt, 0.6, step_image_pd, 1, 0)
                else:
                    step_image_pd = step_image_gt.copy()
                    step_image_mix = step_image_gt.copy()
                buffer_video[f"{video_base_name}-pd.mp4"][0].append(step_image_pd)
                buffer_video[f"{video_base_name}-mix.mp4"][0].append(step_image_mix)

            # ! save agent-centric view
            if save_agent_view:
                for k, v in buffer_video.items():
                    ag_idx = v[1]
                    if ag_idx is not None:
                        text_valid = True
                        if t_pred < 0 or prediction is None:  # draw ground truth when prediction is not available
                            pred_started, t_valid = False, t
                            if not episode["agent/valid"][ag_idx, t_valid]:  # get the first valid step
                                text_valid = False
                                t_valid = np.where(episode["agent/valid"][ag_idx])[0][0]
                            ev_loc = self._to_pixel(episode["agent/pos"][ag_idx, t_valid])
                            ev_rot = episode["agent/yaw_bbox"][ag_idx, t_valid, 0]
                        else:
                            pred_started, t_valid = True, t_pred
                            if not prediction["agent/valid"][ag_idx, t_valid]:  # get the closest valid step
                                text_valid = False
                                valid_steps = np.where(prediction["agent/valid"][ag_idx])[0]
                                t_valid_idx = np.abs(t_valid - valid_steps).argmin()
                                t_valid = valid_steps[t_valid_idx]
                            ev_loc = self._to_pixel(prediction["agent/pos"][ag_idx, t_valid])
                            ev_rot = prediction["agent/yaw_bbox"][ag_idx, t_valid, 0]

                        agent_view = step_image_mix.copy()
                        if not episode["agent/role"][ag_idx].any():
                            color = COLOR_ALUMINIUM_0
                        else:
                            color = self.agent_role_style[np.where(episode["agent/role"][ag_idx])[0].min()]
                        # draw ground-truth dest
                        if "agent/dest" in episode:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(episode["map/pos"][episode["agent/dest"][ag_idx], 0]),
                                color=COLOR_BUTTER_0,
                                thickness=4,
                                line_type=cv2.LINE_AA,
                                tipLength=0.05,
                            )
                        # draw ground-truth goal
                        if "agent/goal" in episode:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(episode["agent/goal"][ag_idx, :2]),
                                color=COLOR_MAGENTA,
                                thickness=4,
                                line_type=cv2.LINE_AA,
                                tipLength=0.05,
                            )
                        # draw predicted dest/goal
                        if "agent/dest" in prediction and prediction["ag_navi_valid"][ag_idx, t_valid]:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(episode["map/pos"][prediction["agent/dest"][ag_idx, t_valid], 0]),
                                color=COLOR_GREEN,
                                thickness=2,
                                line_type=cv2.LINE_AA,
                                tipLength=0.1,
                            )
                        elif "agent/goal" in prediction and prediction["ag_navi_valid"][ag_idx, t_valid]:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(prediction["agent/goal"][ag_idx, t_valid, :2]),
                                color=COLOR_GREEN,
                                thickness=2,
                                line_type=cv2.LINE_AA,
                                tipLength=0.1,
                            )
                        trans = self._get_warp_transform(ev_loc, ev_rot)
                        agent_view = cv2.warpAffine(agent_view, trans, (self.video_size, self.video_size))
                        agent_view = self._add_txt(
                            agent_view, episode, prediction, t_valid, ag_idx, pred_started, text_valid
                        )
                        v[0].append(agent_view)

        for k, v in buffer_video.items():
            encoder = ImageEncoder(k, v[0][0].shape, 20, 20)
            for im in v[0]:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None
        return list(buffer_video.keys())

    @staticmethod
    def _add_txt(
        im: np.ndarray,
        episode: Dict[str, np.ndarray],
        prediction: Optional[Dict[str, np.ndarray]],
        t: int,
        idx: int,
        pred_started: bool,
        text_valid: bool,
        line_width: int = 30,
    ) -> np.ndarray:
        h, w, _ = im.shape
        agent_view = np.zeros([h, w + 200, 3], dtype=im.dtype)
        agent_view[:h, :w] = im
        if (prediction is not None) and pred_started:
            txt_list = [
                f'id:{int(episode["episode_idx"])}',
                f"valid:{int(text_valid)}",
                f'nav_valid:{int(prediction["ag_navi_valid"][idx, t])}',
                f'nav_reach:{int(prediction["navi_reached"][idx, t])}',
                f'out:{int(prediction["outside_map_this_step"][idx, t])}/{int(prediction["outside_map"][idx, t])}',
                f'col:{int(prediction["collided_this_step"][idx, t])}/{int(prediction["collided"][idx, t])}',
                f'col_way:{int(prediction["collided_wosac_this_step"][idx, t])}/{int(prediction["collided_wosac"][idx, t])}',
                f'red:{int(prediction["run_red_light_this_step"][idx, t])}/{int(prediction["run_red_light"][idx, t])}',
                f'edge:{int(prediction["run_road_edge_this_step"][idx, t])}/{int(prediction["run_road_edge"][idx, t])}',
                f'passive:{int(prediction["passive_this_step"][idx, t])}/{int(prediction["passive"][idx, t])}',
                f'r_goal:{int(prediction["goal_reached_this_step"][idx, t])}/{int(prediction["goal_reached"][idx, t])}',
                f'r_dest:{int(prediction["dest_reached_this_step"][idx, t])}/{int(prediction["dest_reached"][idx, t])}',
                # f'x:{prediction["agent/pos"][idx, t, 0]:.2f}',
                # f'y:{prediction["agent/pos"][idx, t, 1]:.2f}',
                # f'yaw:{prediction["agent/yaw_bbox"][idx, t, 0]:.2f}',
                # f'spd:{prediction["motion"][idx, t, 0]:.2f}',
                # f'acc:{prediction["motion"][idx, t, 1]:.2f}',
                # f'yaw_r:{prediction["motion"][idx, t, 2]:.2f}',
                f'acc:{prediction["action"][idx, t, 0]:.2f}',
                f'steer:{prediction["action"][idx, t, 1]:.2f}',
                f'score:{prediction["score"][idx]:.2f}',
                f'act_P:{prediction["act_P"][idx, t]:.2f}',
                f"yellow:gt dest",
                f"magent:gt goal",
                f"green:pred",
            ]
            # differentiable rewards
            if "diffbar_reward_valid" in prediction:
                txt_list.append(f'dr_valid:{int(prediction["diffbar_reward_valid"][idx, t])}')
                txt_list.append(f'dr:{prediction["diffbar_reward"][idx, t]:.2f}')
                txt_list.append(f'il_pos:{prediction["r_imitation_pos"][idx, t]:.2f}')
                txt_list.append(f'il_rot:{prediction["r_imitation_rot"][idx, t]:.2f}')
                txt_list.append(f'il_spd:{prediction["r_imitation_spd"][idx, t]:.2f}')
                txt_list.append(f'rule_apx:{prediction["r_traffic_rule_approx"][idx, t]:.2f}')
        else:
            txt_list = [
                f'id:{int(episode["episode_idx"])}',
                f"valid:{int(text_valid)}",
                f'x:{episode["agent/pos"][idx, t, 0]:.2f}',
                f'y:{episode["agent/pos"][idx, t, 1]:.2f}',
                f'yaw:{episode["agent/yaw_bbox"][idx, t, 0]:.2f}',
                f'spd:{episode["agent/spd"][idx, t, 0]:.2f}',
                f'role:{list(np.where(episode["agent/role"][idx])[0])}',
                f'size_x:{episode["agent/size"][idx, 0]:.2f}',
                f'size_y:{episode["agent/size"][idx, 1]:.2f}',
                f'size_z:{episode["agent/size"][idx, 2]:.2f}',
            ]

        for i, txt in enumerate(txt_list):
            agent_view = cv2.putText(
                agent_view, txt, (w, line_width * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
            )
        return agent_view

    def _to_pixel(self, pos: np.ndarray) -> np.ndarray:
        pos = pos * self.px_per_m
        pos[..., 0] = pos[..., 0] - self.top_left_px[0]
        pos[..., 1] = -pos[..., 1] - self.top_left_px[1]
        return np.round(pos).astype(np.int32)

    def _get_warp_transform(self, loc: np.ndarray, yaw: float) -> np.ndarray:
        """
        loc: xy in pixel
        yaw: in rad
        """

        forward_vec = np.array([np.cos(yaw), -np.sin(yaw)])
        right_vec = np.array([np.sin(yaw), np.cos(yaw)])

        bottom_left = loc - self.px_agent2bottom * forward_vec - (0.5 * self.video_size) * right_vec
        top_left = loc + (self.video_size - self.px_agent2bottom) * forward_vec - (0.5 * self.video_size) * right_vec
        top_right = loc + (self.video_size - self.px_agent2bottom) * forward_vec + (0.5 * self.video_size) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self.video_size - 1], [0, 0], [self.video_size - 1, 0]], dtype=np.float32)
        return cv2.getAffineTransform(src_pts, dst_pts)

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray, agent_pos: np.ndarray, agent_yaw: np.ndarray, agent_size: np.ndarray
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # n, 1
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)  # n,2
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward  # [n, 2]
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right  # [n, 2]

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )  # n,4,2

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset  # n,4,2
        return bbox

    def get_dest_prob_image(
        self, im_base_name: str, episode: Dict[str, np.ndarray], dest_prob: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            episode: Dict for ground truth, for t = {0, ..., prediction["step_gt"]}, 11 steps for test, 91 for train/val
                "agent/valid": [n_ag, n_step/n_step_hist], bool
                "agent/pos": [n_ag, n_step/n_step_hist, 2], (x,y)
                "agent/yaw_bbox": [n_ag, n_step/n_step_hist, 1], [-pi, pi]
                "agent/role": [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
                "agent/size": [n_ag, 3], [length, width, height]
                "map/valid": [n_mp, n_mp_pl_node]
                "map/pos": [n_mp, n_mp_pl_node, 2], (x,y)
                "agent/dest": not exist or [n_ag], int64: index to map [0, n_mp]
            dest_prob: [n_ag, n_mp] float prob
        """
        list_im_idx = {}
        list_im_idx[f"{im_base_name}-sdc.jpg"] = 0
        for i in np.where(episode["agent/role"][:, 1])[0]:
            list_im_idx[f"{im_base_name}-int_{i}.jpg"] = i
        for i in np.where(episode["agent/role"][:, 2])[0]:
            list_im_idx[f"{im_base_name}-pre_{i}.jpg"] = i
        n_others_to_vis = 5
        idx_others = np.where(episode["agent/valid"].any(1) & ~(episode["agent/role"].any(-1)))[0]
        for i in idx_others[:n_others_to_vis]:
            list_im_idx[f"{im_base_name}-other_{i}.jpg"] = i

        for im_path, i in list_im_idx.items():
            t = episode["agent/valid"][i].argmax()
            dest_valid = dest_prob[i] > 1e-4
            p_normalized = dest_prob[i, dest_valid].copy()
            p_max = p_normalized.max()
            p_min = p_normalized.min()
            p_normalized = (p_normalized - p_min) / (p_max - p_min + 1e-4) * 3.0
            m_type = np.zeros_like(episode["map/type"][dest_valid])
            m_type[:, 1] = True
            for k in p_normalized.argsort()[-6:]:
                m_type[k, 1] = False
                m_type[k, 3] = True
            im = self._draw_map(
                raster_map=np.zeros_like(self.raster_map),
                map_valid=episode["map/valid"][dest_valid],
                map_type=m_type,
                map_pos=episode["map/pos"][dest_valid],
                attn_weights_to_pl=p_normalized,
            )
            # ! draw gt_dest if available
            if "agent/dest" in episode:
                draw_gt_dest = self._to_pixel(episode["map/pos"][episode["agent/dest"][i]])
                # [n_pl, 20]
                draw_gt_dest_valid = episode["map/valid"][episode["agent/dest"][i]]
                cv2.polylines(
                    im,
                    [draw_gt_dest[draw_gt_dest_valid]],
                    isClosed=False,
                    color=COLOR_MAGENTA,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            ag_pos = episode["agent/pos"][i, t]
            ag_yaw_bbox = episode["agent/yaw_bbox"][i, t]
            bbox_gt = self._get_agent_bbox(episode["agent/valid"][i, t], ag_pos, ag_yaw_bbox, episode["agent/size"][i])
            bbox_gt = self._to_pixel(bbox_gt)
            heading_start = self._to_pixel(ag_pos)
            heading_end = self._to_pixel(
                ag_pos + 1.5 * np.stack([np.cos(ag_yaw_bbox[0]), np.sin(ag_yaw_bbox[0])], axis=-1)
            )
            cv2.fillConvexPoly(im, bbox_gt, color=COLOR_RED)
            cv2.arrowedLine(
                im, heading_start, heading_end, color=COLOR_BLACK, thickness=4, line_type=cv2.LINE_AA, tipLength=0.6
            )
            cv2.imwrite(im_path, im[..., ::-1])
        return list(list_im_idx.keys())
