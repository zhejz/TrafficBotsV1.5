import torch
from torch import Tensor
from typing import Tuple


# Constant distance to apply when distances between objects are invalid. This
# will avoid the propagation of nans and should be reduced out when taking the
# minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Collision threshold, i.e. largest distance between objects that is considered
# to be a collision.
COLLISION_DISTANCE_THRESHOLD = 0.0
# Rounding factor to apply to the corners of the object boxes in distance and
# collision computation. The rounding factor is between 0 and 1, where 0 yields
# rectangles with sharp corners (no rounding) and 1 yields capsule shapes.
# Default value of 0.7 conservately fits most vehicle contours.
CORNER_ROUNDING_FACTOR = 0.7


def get_ag_bbox(pose: Tensor, ag_size: Tensor) -> Tensor:
    """
    Args:
        pose: [n_sc, n_ag, 3] (x,y,yaw)
        ag_size: [n_sc, n_ag, 2], length, width

    Returns:
        ag_bbox: [n_sc, n_ag, 4, 2]
    """
    heading_cos = torch.cos(pose[..., 2])  # [n_sc, n_ag]
    heading_sin = torch.sin(pose[..., 2])  # [n_sc, n_ag]
    heading_f = torch.stack([heading_cos, heading_sin], axis=-1)  # [n_sc, n_ag, 2]
    heading_r = torch.stack([heading_sin, -heading_cos], axis=-1)  # [n_sc, n_ag, 2]
    offset_forward = 0.5 * ag_size[..., [0]].expand(-1, -1, 2) * heading_f  # [n_sc, n_ag, 2]
    offset_right = 0.5 * ag_size[..., [1]].expand(-1, -1, 2) * heading_r  # [n_sc, n_ag, 2]
    vertex_offset = torch.stack(
        [
            offset_forward - offset_right,
            -offset_forward - offset_right,
            -offset_forward + offset_right,
            offset_forward + offset_right,
        ],
        dim=2,
    )
    ag_bbox = pose[:, :, None, :2].expand(-1, -1, 4, -1) + vertex_offset
    return ag_bbox


def _signed_distance_from_point_to_convex_polygon(query_points: Tensor, polygon_points: Tensor) -> Tensor:
    """Finds the signed distances from query points to convex polygons.

    Each polygon is represented by a 2d tensor storing the coordinates of its
    vertices. The vertices must be ordered in counter-clockwise order. An
    arbitrary number of pairs (point, polygon) can be batched on the 1st
    dimension.

    Note: Each polygon is associated to a single query point.

    Args:
        query_points: [n_sc, n_polygon, 2]
            The last dimension is the x and y coordinates of points.
        polygon_points: [n_sc, n_polygon, n_point, 2]
            The last dimension is the x and y coordinates of vertices.

    Returns:
        [n_sc, n_polygon]. A tensor containing the signed distances of the query points to the polygons.
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(polygon_points)
    # [n_sc, n_polygon, n_point, 2], [n_sc, n_polygon, n_point, 2], [n_sc, n_polygon, n_point]

    # [n_sc, n_polygon, n_point, 2]
    vertices_to_query_vectors = query_points.unsqueeze(2) - polygon_points
    # Shape (num_polygons, num_points_per_polygon).
    vertices_distances = torch.norm(vertices_to_query_vectors, dim=-1)

    # Query point to edge distances are measured as the perpendicular distance
    # of the point from the edge. If the projection of this point on to the edge
    # falls outside the edge itself, this distance is not considered (as there)
    # will be a lower distance with the vertices of this specific edge.

    # Make distances negative if the query point is in the inward side of the
    # edge. Shape: (num_polygons, num_points_per_polygon).
    edge_signed_perp_distances = torch.sum(-normal_unit_vectors * vertices_to_query_vectors, dim=-1)

    # If `edge_signed_perp_distances` are all less than 0 for a
    # polygon-query_point pair, then the query point is inside the convex polygon.
    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=-1)

    # Project the distances over the tangents of the edge, and verify where the
    # projections fall on the edge.
    # Shape: (num_polygons, num_edges_per_polygon).
    projection_along_tangent = torch.sum(tangent_unit_vectors * vertices_to_query_vectors, dim=-1)
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths
    # Shape: (num_polygons, num_edges_per_polygon).
    is_projection_on_edge = torch.logical_and(
        projection_along_tangent_proportion >= 0.0, projection_along_tangent_proportion <= 1.0
    )

    # If the point projection doesn't lay on the edge, set the distance to inf.
    edge_perp_distances = torch.abs(edge_signed_perp_distances)
    edge_distances = torch.where(
        is_projection_on_edge, edge_perp_distances, torch.zeros_like(edge_perp_distances) + EXTREMELY_LARGE_DISTANCE
    )

    # Aggregate vertex and edge distances.
    # Shape: (num_polyons, 2 * num_edges_per_polygon).
    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=-1)
    # Aggregate distances per polygon and change the sign if the point lays inside
    # the polygon. Shape: (num_polygons,).
    min_distance = torch.amin(edge_and_vertex_distance, dim=-1)
    signed_distances = torch.where(is_inside, -min_distance, min_distance)
    return signed_distances


def _get_edge_info(polygon_points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes properties about the edges of a polygon.

        Args:
            polygon_points: vertices of each polygon, [n_sc, n_polygon, n_point, 2].
            Each polygon is assumed to have an equal number of vertices.

        Returns:
            tangent_unit_vectors: [n_sc, n_polygon, n_point, 2]
                A unit vector in (x,y) with the same direction as the tangent to the edge. Shape
            normal_unit_vectors: [n_sc, n_polygon, n_point, 2]
                A unit vector in (x,y) with the same direction as the normal to the edge.
            edge_lengths: [n_sc, n_polygon, n_point] Lengths of the edges.
    """
    shifted_polygon_points = polygon_points.roll(-1, dims=2)
    edge_vectors = shifted_polygon_points - polygon_points
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    normal_unit_vectors = torch.stack([-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1)
    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def _get_downmost_edge_in_box(box: Tensor) -> Tuple[Tensor, Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

        Note: We assume box edges are given in a counter-clockwise order, so that
        the edge which starts with the downmost vertex (i.e. the downmost edge) is
        uniquely identified.

        Args:
            box: [n_sc, n_box, 4, 2]. The last dimension contains the x-y coordinates of corners in boxes.

        Returns:
            downmost_vertex_idx: [n_sc, n_box, 1]
                The index of the downmost vertex, which is also the index of the downmost edge.
            downmost_edge_direction: [n_sc, n_box, 1, 2]
                The tangent unit vector of the downmost edge, pointing in the counter-clockwise direction of the box.
    """
    # The downmost vertex is the lowest in the y dimension.
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1).unsqueeze(-1)  # [n_sc, n_box, 1]

    # Find the counter-clockwise point edge from the downmost vertex.
    _idx_sc = torch.arange(box.shape[0])[:, None, None]  # [n_sc, 1, 1]
    _idx_box = torch.arange(box.shape[1])[None, :, None]  # [1, n_box, 1]
    edge_start_vertex = box[_idx_sc, _idx_box, downmost_vertex_idx]  # [n_sc, n_boxes, 1, 2]
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, 4)
    edge_end_vertex = box[_idx_sc, _idx_box, edge_end_idx]  # [n_sc, n_boxes, 1, 2]

    # Compute the direction of this downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(-1)
    return downmost_vertex_idx, downmost_edge_direction


def _minkowski_sum_of_box_and_box_points(box1_points: Tensor, box2_points: Tensor) -> Tensor:
    _idx_sc = torch.arange(box1_points.shape[0])[:, None, None]  # [n_sc, 1, 1]
    _idx_box = torch.arange(box1_points.shape[1])[None, :, None]  # [1, n_box, 1]
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=_idx_sc.dtype, device=box1_points.device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=_idx_sc.dtype, device=box1_points.device)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(box2_points)

    condition = (
        downmost_box1_edge_direction[..., 0] * downmost_box2_edge_direction[..., 1]
        - downmost_box1_edge_direction[..., 1] * downmost_box2_edge_direction[..., 0]
    ) >= 0.0
    condition = condition.expand(-1, -1, 8)  # [n_sc, n_box, 8]

    box1_point_order = torch.where(condition, point_order_2, point_order_1)  # [n_sc, n_box, 8]
    box1_point_order = torch.remainder(box1_point_order + box1_start_idx, 4)  # [n_sc, n_box, 8]
    ordered_box1_points = box1_points[_idx_sc, _idx_box, box1_point_order]  # [n_sc, n_box, 8, 2]

    # Gather points from box2 as well.
    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = torch.remainder(box2_point_order + box2_start_idx, 4)
    ordered_box2_points = box2_points[_idx_sc, _idx_box, box2_point_order]
    minkowski_sum = ordered_box1_points + ordered_box2_points
    return minkowski_sum


def check_collided_wosac(
    pose: Tensor,  # [n_sc, n_ag, 3], x,y,yaw
    ag_size: Tensor,  # [n_sc, n_ag, 3], length, width, height
    valid: Tensor,  # [n_sc, n_ag] bool
) -> Tensor:  # [n_sc, n_ag] bool
    """
    Args:
        evaluated_object_mask: A boolean tensor of shape (num_objects), to index the
        objects identified by the tensors defined above. If True, the object is
        considered part of the "evaluation set", i.e. the object can actively
        collide into other objects. If False, the object can also be passively
        collided into.

    Returns:
        A tensor of shape (num_evaluated_objects, num_steps), containing the
        distance to the nearest object, for each timestep and for all the objects
        to be evaluated, as specified by `evaluated_object_mask`.
  """
    n_sc, n_ag, _ = pose.shape
    shrinking_distance = torch.min(ag_size[:, :, 0], ag_size[:, :, 1]) * CORNER_ROUNDING_FACTOR / 2.0  # [n_sc, n_ag]

    box_corners = get_ag_bbox(pose, ag_size[:, :, :2] - 2.0 * shrinking_distance.unsqueeze(-1))  # [n_sc, n_ag, 4, 2]

    # [n_sc, n_ag*n_ag, 4, 2]
    eval_corners = box_corners.unsqueeze(2).expand(-1, -1, n_ag, -1, -1).flatten(1, 2)
    all_corners = box_corners.unsqueeze(1).expand(-1, n_ag, -1, -1, -1).flatten(1, 2)

    # [n_sc, n_ag*n_ag, 8, 2]
    minkowski_sum = _minkowski_sum_of_box_and_box_points(eval_corners, -1.0 * all_corners)

    # If the two convex shapes intersect, the Minkowski subtraction polygon will containing the origin.
    signed_distances_flat = _signed_distance_from_point_to_convex_polygon(
        query_points=torch.zeros_like(minkowski_sum[:, :, 0, :]), polygon_points=minkowski_sum
    )  # [n_sc, n_ag*n_ag]

    signed_distances = signed_distances_flat.view(n_sc, n_ag, n_ag)

    signed_distances -= shrinking_distance.unsqueeze(1)
    signed_distances -= shrinking_distance.unsqueeze(2)

    invalid_mask = ~(valid.unsqueeze(1) & valid.unsqueeze(2))  # [n_sc, n_ag, n_ag]
    invalid_mask |= torch.eye(n_ag, dtype=invalid_mask.dtype, device=invalid_mask.device).unsqueeze(0)
    signed_distances.masked_fill_(invalid_mask, EXTREMELY_LARGE_DISTANCE)
    return torch.amin(signed_distances, dim=2) < COLLISION_DISTANCE_THRESHOLD

