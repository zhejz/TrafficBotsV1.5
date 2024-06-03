# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_rad2local


@torch.no_grad()
def get_rel_pose(
    pose: Tensor, invalid: Tensor, pose2: Optional[Tensor] = None, invalid2: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        pose: [n_sc, n_src, 3], (x,y,yaw), in global coordinate
        invalid: [n_sc, n_src]
        pose2: [n_sc, n_tgt, 3], (x,y,yaw), in global coordinate, or None
        invalid2: [n_sc, n_tgt]

    Returns:
        rel_pose: [n_sc, n_src, n_src/n_tgt, 3] (x,y,yaw)
        rel_dist: [n_sc, n_src, n_src/n_tgt]
    """
    if pose2 is None:
        pose2, invalid2 = pose, invalid

    xy, yaw = pose[:, :, :2], pose[:, :, -1]  # [n_sc, n_src, 2], [n_sc, n_src]
    xy2, yaw2 = pose2[:, :, :2], pose2[:, :, -1]  # [n_sc, n_tgt, 2], [n_sc, n_tgt]
    rel_pose = torch.cat(
        [
            torch_pos2local(xy2.unsqueeze(1), xy.unsqueeze(2), torch_rad2rot(yaw)),
            torch_rad2local(yaw2.unsqueeze(1), yaw, cast=False).unsqueeze(-1),
        ],
        dim=-1,
    )  # [n_sc, n_src, n_src, 3]
    rel_dist = torch.norm(rel_pose[..., :2], dim=-1)  # [n_sc, n_src, n_tgt]
    rel_dist.masked_fill_(invalid.unsqueeze(2) | invalid2.unsqueeze(1), float("inf"))
    return rel_pose, rel_dist


@torch.no_grad()
def get_rel_dist(
    xy: Tensor, invalid: Tensor, xy2: Optional[Tensor] = None, invalid2: Optional[Tensor] = None
) -> Tensor:
    """
    Args:
        xy: [n_sc, n_src, 2], in global coordinate
        invalid: [n_sc, n_src]
        xy2: [n_sc, n_tgt, 2], in global coordinate, or None
        invalid2: [n_sc, n_tgt]

    Returns:
        rel_dist: [n_sc, n_src, n_src/n_tgt]
    """
    if xy2 is None:
        xy2, invalid2 = xy, invalid
    rel_dist = torch.norm(xy.unsqueeze(2) - xy2.unsqueeze(1), dim=-1)  # [n_sc, n_src, n_tgt]
    rel_dist.masked_fill_(invalid.unsqueeze(2) | invalid2.unsqueeze(1), float("inf"))
    return rel_dist


@torch.no_grad()
def get_tgt_knn_idx(
    tgt_invalid: Tensor, rel_pose: Optional[Tensor], rel_dist: Tensor, n_tgt_knn: int, dist_limit: Union[float, Tensor],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Args:
        tgt_invalid: [n_sc, n_tgt]
        rel_pose: [n_sc, n_src, n_tgt, 3] or None
        rel_dist: [n_sc, n_src, n_tgt]
        knn: int, set to <=0 to skip knn, i.e. n_tgt_knn=n_tgt
        dist_limit: float, or Tensor [n_sc, n_tgt, 1]

    Returns:
        idx_tgt: [n_sc, n_src, n_tgt_knn]
        tgt_invalid_knn: [n_sc, n_src, n_tgt_knn]
        rpe: [n_sc, n_src, n_tgt_knn, 3] or None
    """
    n_sc, n_src, n_tgt = rel_dist.shape
    assert 0 < n_tgt_knn < n_tgt

    idx_scene = torch.arange(n_sc)[:, None, None]  # [n_sc, 1, 1]
    idx_src = torch.arange(n_src)[None, :, None]  # [1, n_src, 1]
    # [n_sc, n_src, n_tgt_knn]
    dist_knn, idx_tgt = torch.topk(rel_dist, n_tgt_knn, dim=-1, largest=False, sorted=False)
    tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)[idx_scene, idx_src, idx_tgt]
    tgt_invalid_knn = tgt_invalid_knn | (dist_knn > dist_limit)

    rpe = None if rel_pose is None else rel_pose[idx_scene, idx_src, idx_tgt]

    return idx_tgt, tgt_invalid_knn, rpe
