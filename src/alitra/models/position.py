from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .frame import Frame


@dataclass
class Position:
    """
    Position contains the x, y and z coordinate as well as a frame
    """

    x: float
    y: float
    z: float
    frame: Frame

    def to_array(self) -> np.ndarray:
        """
        :return: Numpy array of position, shape (3,)
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(position: np.ndarray, frame: Frame) -> Position:
        """
        :param position: Numpy array of position [x,y,z].
            Needs to be shape (3,)
        :param frame: Frame of position
        """
        if position.shape != (3,):
            raise ValueError("position array must have shape (3,)")
        return Position(x=position[0], y=position[1], z=position[2], frame=frame)


@dataclass
class Positions:
    """
    Positions contains a list of positions as well as a frame in which the points
    are valid
    """

    positions: List[Position]
    frame: Frame

    def to_array(self) -> np.ndarray:
        """
        :return: Numpy array of positions, shape (N,3)
        """
        positions = []
        for position in self.positions:
            positions.append([position.x, position.y, position.z])
        return np.array(positions, dtype=float)

    @staticmethod
    def from_array(position_array: np.ndarray, frame: Frame) -> Positions:
        """
        :param position_array: Numpy array of positions i.e. [[x,y,z],[x,y,z]].
            Needs to be shape (N,3)
        :param frame: Frame of positions
        """
        if len(position_array.shape) < 2 or position_array.shape[1] != 3:
            raise ValueError("position_array should have shape (N,3)")
        positions: List[Position] = []
        for position in position_array:
            positions.append(
                Position(x=position[0], y=position[1], z=position[2], frame=frame)
            )
        return Positions(positions=positions, frame=frame)

    def __str__(self):
        """
        :return: Unique string representation of the position, ignoring the frame
        """
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"
