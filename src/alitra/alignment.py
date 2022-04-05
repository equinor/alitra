from typing import Literal, Tuple

import numpy as np
from numpy.linalg import norm  # type: ignore
from scipy.spatial.transform import Rotation

from .models.map import Map
from .models.position import Positions
from .models.translation import Translation
from .transform import Transform


def align_maps(
    map_from: Map,
    map_to: Map,
    rot_axes: Literal["x", "y", "z", "xyz"],
    rsmd_threshold=0.4,
) -> Transform:
    """
    Uses align_positions to create a transform between two maps.
    See align_positions for further information.

    :param map_from: Map the transform is coming from
    :param map_to: Map the transform is going to
    :param rot_axes: Axis of rotation. For rotations in the xy plane (most common),
        this is set to 'z'
    :param rsmd_threshold: The root mean square distance threshold, for the coordinate
        fitting error in matching the two coordinate systems.
    """
    return align_positions(
        map_from.reference_positions,
        map_to.reference_positions,
        rot_axes,
        rsmd_threshold,
    )


def align_positions(
    positions_from: Positions,
    positions_to: Positions,
    rot_axes: Literal["x", "y", "z", "xyz"],
    rsmd_threshold=0.4,
) -> Transform:
    """
    Let positions_from be fixed local coordinate frame, and positions_to be some other
    fixed coordinate system. Further, let the relationship between the two reference
    systems be described as

    positions_from = rotation.apply(positions_to) + translation

    This function finds the rotation and translation by matching the two
    coordinate systems, and represent the transformation through a Transform object.
    For robustness it is advised to use more than 2 positions in alignment and using
    positions with some distance to each other.
    :param positions_from: Coordinates in a fixed frame
    :param positions_to: Coordinates in a fixed frame
    :param rot_axes: Axis of rotation. For rotations in the xy plane (most common),
        this is set to 'z'
    :param rsmd_threshold: The root mean square distance threshold, for the coordinate
        fitting error in matching the two coordinate systems.
    """
    if len(positions_from.positions) != len(positions_to.positions):
        raise ValueError(
            f"Expected inputs 'positions_from' and 'positions_to' to have the same shapes"
            + f" got {len(positions_from.positions)} and {len(positions_to.positions)}, respectively"
        )
    if len(positions_from.positions) < 2:
        raise ValueError(
            f" Expected at least 2 positions, got {len(positions_from.positions)}"
        )
    if len(positions_from.positions) < 3 and rot_axes == "xyz":
        raise ValueError(
            f" Expected at least 3 positions, got {len(positions_from.positions)}"
        )

    try:
        edges_1, edges_2 = _get_edges(positions_from, positions_to, rot_axes)
    except Exception as e:
        raise ValueError(e)

    rotation, rmsd_rot, sensitivity = Rotation.align_vectors(
        edges_2, edges_1, return_sensitivity=True
    )

    translations: Translation = Translation.from_array(
        np.mean(
            positions_to.to_array() - rotation.apply(positions_from.to_array()),
            axis=0,  # type:ignore
        ),
        from_=positions_from.frame,
        to_=positions_to.frame,
    )  # type: ignore
    transform = Transform(
        from_=positions_from.frame,
        to_=positions_to.frame,
        translation=translations,
        rotation=rotation,
    )

    try:
        _check_rsme_treshold(transform, positions_to, positions_from, rsmd_threshold)
    except Exception as e:
        raise ValueError(e)

    return transform


def _get_edges(
    positions_from: Positions,
    positions_to: Positions,
    rot_axes: Literal["x", "y", "z", "xyz"],
    tol: float = 10e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    edges_from = _get_edges_between_coordinates(positions_from)
    edges_to = _get_edges_between_coordinates(positions_to)
    if np.min(norm((np.vstack([edges_from, edges_to])), axis=1)) < tol:
        raise ValueError("Positions are not unique")
    edges_from, edges_to = _add_dummy_rot_axis_edge(edges_from, edges_to, rot_axes)
    return edges_from, edges_to


def _get_edges_between_coordinates(positions_from: Positions) -> np.ndarray:
    """Finds all edges (vectors) between the input coordinates"""
    positions_from_arr = positions_from.to_array()
    n_positions = positions_from_arr.shape[0]
    edges = np.empty([sum(range(n_positions)), 3])
    index = 0
    for i in range(0, n_positions - 1):
        for j in range(i + 1, n_positions):
            edges[index, :] = np.array(
                [
                    positions_from_arr[i] - positions_from_arr[j],
                ]
            )
            index = index + 1
    return edges


def _add_dummy_rot_axis_edge(
    edges_from: np.ndarray,
    edges_to: np.ndarray,
    rot_axes: Literal["x", "y", "z", "xyz"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Adds vectors to help ensure no rotations about non specified axes"""
    if rot_axes == "z":
        edges_from[:, 2] = 0
        edges_to[:, 2] = 0
        edges_from = np.vstack([edges_from, np.array([[0, 0, 1]])])
        edges_to = np.vstack([edges_to, np.array([[0, 0, 1]])])
    if rot_axes == "y":
        edges_from[:, 1] = 0
        edges_to[:, 1] = 0
        edges_from = np.vstack([edges_from, np.array([[0, 1, 0]])])
        edges_to = np.vstack([edges_to, np.array([[0, 1, 0]])])
    if rot_axes == "x":
        edges_from[:, 0] = 0
        edges_to[:, 0] = 0
        edges_from = np.vstack([edges_from, np.array([[1, 0, 0]])])
        edges_to = np.vstack([edges_to, np.array([[1, 0, 0]])])
    return edges_from, edges_to


def _check_rsme_treshold(
    transform: Transform,
    positions_to: Positions,
    positions_from: Positions,
    rsmd_threshold: float,
) -> float:
    positions_to_new = transform.transform_position(
        positions_to, from_=positions_to.frame, to_=positions_from.frame
    )
    transform_distance_error = positions_from.to_array() - positions_to_new.to_array()
    rsm_distance = np.mean(norm(transform_distance_error, axis=1))
    if rsm_distance > rsmd_threshold:
        raise ValueError(
            f"Root mean square error {rsm_distance:.4f} exceeds treshold {rsmd_threshold}"
        )
    return rsmd_threshold
