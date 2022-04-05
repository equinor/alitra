from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .frame import Frame
from .orientation import Orientation
from .position import Position


@dataclass
class Pose:
    """
    Pose contains a position, an orientation and a frame
    """

    position: Position
    orientation: Orientation
    frame: Frame

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Tuple of numpy arrays of the position and orientation as a quaternion
            with shapes (3,) and (4,) respectively
        """
        return self.position.to_array(), self.orientation.to_quat_array()

    @staticmethod
    def from_array(pos_array: np.ndarray, quat_array: np.ndarray, frame: Frame) -> Pose:
        """
        :param pos_array: Numpy array of shape (3,) containing position [x,y,z]
        :param quat_array: Numpy array of shape (4,) containing orientation as a
            quaternion [x,y,z,w]
        :param frame: Frame of pose
        :return: Pose object
        """
        return Pose(
            Position.from_array(pos_array, frame),
            Orientation.from_quat_array(quat_array, frame),
            frame,
        )
