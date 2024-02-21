from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .frame import Frame


@dataclass
class Orientation:
    """
    This class represents an orientation using the quaternion values:
    x, y, z, w, and a frame. This classes uses scipy rotation to handle transformations
    between euler and quaternions
    """

    x: float
    y: float
    z: float
    w: float
    frame: Frame

    def to_euler_array(
        self, degrees: bool = False, wrap_angles: bool = False, seq: str = "ZYX"
    ) -> np.ndarray:
        """
        :param degrees: Set to true to retrieve angles as degrees
        :param wrap_angles: Set to true to get angles between 0 and 360 deg or
            0 and two pi
        :param seq: Sequence of axes for rotations, same as scipy rotation
        :return: Numpy array of euler angles
        """
        rotation: Rotation = Rotation.from_quat(self.to_quat_array())
        euler = rotation.as_euler(seq=seq, degrees=degrees)

        if wrap_angles:
            base = 360.0 if degrees else 2 * np.pi
            euler = np.fromiter(map(lambda angle: (angle % base), euler), dtype=float)

        return euler

    def to_quat_array(self) -> np.ndarray:
        """
        :return: Numpy array of quaternion values, [x,y,z,w]
        """
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    def to_rotation(self) -> Rotation:
        """
        :return: Scipy Rotation object
        """
        return Rotation.from_quat(self.to_quat_array())

    @staticmethod
    def from_quat_array(quat: np.ndarray, frame: Frame) -> Orientation:
        """
        :param quat: Numpy array of shape (4,) containing quaternion values, [x,y,z,w]
        :param frame: Frame of orientation
        :return: Orientation object
        """
        if quat.shape != (4,):
            raise ValueError("quaternion should have shape (4,)")
        return Orientation(
            *quat,  # type: ignore
            frame=frame,
        )

    @staticmethod
    def from_euler_array(
        euler: np.ndarray, frame: Frame, degrees: bool = False, seq: str = "ZYX"
    ) -> Orientation:
        """
        :param euler: Numpy array of euler angles of shape (3,)
        :param frame: Frame of orientation
        :param degrees: Set to true to retrieve angles as degrees
        :param wrap_angles: Set to true to get angles between 0 and 360 deg or
            0 and two pi
        :param seq: Sequence of axes for rotations, same as scipy rotation
        :return: Orientation object
        """
        rotation = Rotation.from_euler(seq=seq, angles=euler, degrees=degrees)
        return Orientation(*rotation.as_quat(), frame=frame)  # type: ignore

    @staticmethod
    def from_rotation(rotation: Rotation, frame: Frame) -> Orientation:
        """
        :param rotation: Scipy Rotation object
        :param frame: Frame of orientation
        :return: Orientation object
        """
        return Orientation(*rotation.as_quat(), frame=frame)  # type: ignore

    def __str__(self):
        """
        :return: Unique string representation of the orientation, ignoring the frame
        """
        return (
            "["
            + str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.z)
            + ","
            + str(self.w)
            + "]"
        )
