from typing import Literal, Tuple

import numpy as np
from numpy.linalg import norm  # type: ignore
from scipy.spatial.transform import Rotation

from alitra.frame_dataclasses import PointList, Translation, Transform
from alitra.frame_transform import FrameTransform


class AlignFrames:
    @staticmethod
    def align_frames(
        p_1: PointList,
        p_2: PointList,
        rot_axes: Literal["x", "y", "z", "xyz"],
        rsmd_treshold=0.4,
    ) -> FrameTransform:
        """Let p_1 be fixed local coordinate frame that the robot operate in, and p_2 be
        in the asset-fixed global coordinate system. Further, let the relationship between the two
        reference systems be described in the _to frame be : p_1 = rotation_object.apply(p_2) + translation,
        This function finds the rotation_object and translation by matching the two coordinate systems,
        and represent the transformation through a FrameTransform object.
        For robustness it is adviced to use more than 2 points in frame alignment and having using points with some
        distance to each other.
        :param p_1: Coordinates in the fixed local reference frame (local robot map).
        :param p_2: Coordinates in the fixed global reference frame (asset system).
        :param rot_axes: Axis of rotation. For rotations in the xy plane (most common), this is set to 'z'
        :param rsmd_treshold: The root mean square distance treshold,
        for the coordinate fitting error in matching the two coordinate systems.
        """
        if len(p_1.points) != len(p_2.points):
            raise ValueError(
                f"Expected inputs 'p_1' and 'p_2' to have the same shapes"
                + f" got {len(p_1.points)} and {len(p_2.points)}, respectively"
            )

        if len(p_1.points) < 2:
            raise ValueError(f" Expected at least 2 points, got {len(p_1.points)}")
        if len(p_1.points) < 3 and rot_axes == "xyz":
            raise ValueError(f" Expected at least 3 points, got {len(p_1.points)}")

        try:
            edges_1, edges_2 = AlignFrames._get_edges(p_1, p_2, rot_axes)
        except Exception as e:
            raise ValueError(e)

        rotation_object, rmsd_rot, sensitivity = Rotation.align_vectors(
            edges_2, edges_1, return_sensitivity=True
        )

        translations: Translation = Translation.from_array(
            np.mean(
                p_2.as_np_array() - rotation_object.apply(p_1.as_np_array()),
                axis=0,  # type:ignore
            ),
            from_=p_1.frame,
            to_=p_2.frame,
        )  # type: ignore
        transform = Transform(
            from_=p_1.frame,
            to_=p_2.frame,
            translation=translations,
            rotation_object=rotation_object,
        )

        try:
            frame_transform: FrameTransform = FrameTransform(transform)
        except ValueError as e:
            raise ValueError(e)
        try:
            AlignFrames._check_rsme_treshold(frame_transform, p_2, p_1, rsmd_treshold)
        except Exception as e:
            raise ValueError(e)

        return frame_transform

    @staticmethod
    def _get_edges(
        p_1: PointList, p_2: PointList, rot_axes: Literal["x", "y", "z", "xyz"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        edges_1 = AlignFrames._get_edges_between_coordinates(p_1)
        edges_2 = AlignFrames._get_edges_between_coordinates(p_2)
        if np.min(norm((np.vstack([edges_1, edges_2])), axis=1)) < 10e-2:
            raise ValueError("Points are not unique")
        edges_1, edges_2 = AlignFrames._add_dummy_rot_axis_edge(
            edges_1, edges_2, rot_axes
        )
        return edges_1, edges_2

    @staticmethod
    def _get_edges_between_coordinates(p_1: PointList) -> np.ndarray:
        """Finds all edged (vectors) between the input coordinates"""
        p_1_arr = p_1.as_np_array()
        n_points = p_1_arr.shape[0]
        edges_1 = np.empty([sum(range(n_points)), 3])
        index = 0
        for i in range(0, n_points - 1):
            for j in range(i + 1, n_points):
                edges_1[index, :] = np.array(
                    [
                        p_1_arr[i] - p_1_arr[j],
                    ]
                )
                index = index + 1
        return edges_1

    @staticmethod
    def _add_dummy_rot_axis_edge(
        edges_1: np.ndarray,
        edges_2: np.ndarray,
        rot_axes: Literal["x", "y", "z", "xyz"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adds vectors to help ensure no rotations about non specified axes"""
        if rot_axes == "z":
            edges_1[:, 2] = 0
            edges_2[:, 2] = 0
            edges_1 = np.vstack([edges_1, np.array([[0, 0, 1]])])
            edges_2 = np.vstack([edges_2, np.array([[0, 0, 1]])])
        if rot_axes == "y":
            edges_1[:, 1] = 0
            edges_2[:, 1] = 0
            edges_1 = np.vstack([edges_1, np.array([[0, 1, 0]])])
            edges_2 = np.vstack([edges_2, np.array([[0, 1, 0]])])
        if rot_axes == "x":
            edges_1[:, 0] = 0
            edges_2[:, 0] = 0
            edges_1 = np.vstack([edges_1, np.array([[1, 0, 0]])])
            edges_2 = np.vstack([edges_2, np.array([[1, 0, 0]])])
        return edges_1, edges_2

    @staticmethod
    def _check_rsme_treshold(
        frame_transform: FrameTransform, p_2, p_1, rsmd_treshold
    ) -> float:
        p_2_to_1 = frame_transform.transform_point(p_2, from_=p_2.frame, to_=p_1.frame)
        transform_distance_error = p_1.as_np_array() - p_2_to_1.as_np_array()
        rsm_distance = np.mean(norm(transform_distance_error, axis=1))
        if rsm_distance > rsmd_treshold:
            raise ValueError(
                f"Root mean square error {rsm_distance:.4f} exceeds treshold {rsmd_treshold}"
            )
        return rsmd_treshold
