from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Transform:
    """
    A transform object that describe the transformation between two frames.
    Euler or quaternion must be provided to perform a rotation. If no rotation is required specify the unit quaternion
    or zero Euler angles. Translations must be expressed in the (to_) frame
    """

    translation: Translation
    from_: Literal["robot", "asset"]
    to_: Literal["asset", "robot"]
    rotation_object: Rotation = None
    euler: Euler = None
    quaternion: Quaternion = None

    def __post_init__(self):
        if (
            not self.translation.from_ == self.from_
            and not self.translation.to_ == self.to_
            or not self.to_ == self.translation.frame
        ):
            raise ValueError(
                f"The from_ frames or to_ frames of translation and transform object are not equal."
            )

        if self.euler is None and self.quaternion is None:
            raise ValueError("Euler or quaternion must be set to describe the rotation")
        elif self.euler is not None and self.quaternion is not None:
            raise ValueError("Euler and quaternion cannot be set at the same time")
        elif self.euler:
            self.rotation_object = Rotation.from_euler(
                "zyx", self.euler.as_np_array(), degrees=False
            )
        elif self.quaternion:
            self.rotation_object = Rotation.from_quat(self.quaternion.as_np_array())


@dataclass
class Point:
    x: float
    y: float
    frame: Literal["robot", "asset"]
    z: float = 0

    def as_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(coordinate: np.ndarray, frame: Literal["robot", "asset"]) -> Point:
        if coordinate.shape != (3,):
            raise ValueError("Point should have shape (3,)")
        return Point(x=coordinate[0], y=coordinate[1], z=coordinate[2], frame=frame)


@dataclass
class PointList:
    points: List[Point]
    frame: Literal["robot", "asset"]

    def as_np_array(self) -> np.ndarray:
        points = []
        for point in self.points:
            points.append([point.x, point.y, point.z])
        return np.array(points, dtype=float)

    @staticmethod
    def from_array(
        point_array: np.ndarray, frame: Literal["robot", "asset"]
    ) -> PointList:
        if point_array.shape[1] != 3:
            raise ValueError("Coordinate_list should have shape (3,N)")
        points: List[Point] = []
        for point in point_array:
            points.append(Point(x=point[0], y=point[1], z=point[2], frame=frame))
        return PointList(points=points, frame=frame)


@dataclass
class Euler:
    """Euler angles where (psi,phi,theta) is rotation about the z-,y-, and x- axis respectively. as_array and from_array
    are both ordered by rotation about z,y,x. That is [psi,phi,theta]"""

    from_: Literal["robot", "asset"]
    to_: Literal["robot", "asset"]
    psi: float = 0
    phi: float = 0
    theta: float = 0

    def as_np_array(self) -> np.ndarray:
        return np.array([self.psi, self.phi, self.theta], dtype=float)

    @staticmethod
    def from_array(
        rotations: np.ndarray,
        from_: Literal["robot", "asset"],
        to_: Literal["robot", "asset"],
    ) -> Euler:
        if rotations.shape != (3,):
            raise ValueError("Coordinate_list should have shape (3,)")
        return Euler(
            psi=rotations[0], phi=rotations[1], theta=rotations[2], from_=from_, to_=to_
        )


@dataclass
class Quaternion:
    frame: Literal["robot", "asset"]
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1

    def as_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    @staticmethod
    def from_array(
        quaternion: np.ndarray,
        frame: Literal["robot", "asset"],
    ) -> Quaternion:
        if quaternion.shape != (4,):
            raise ValueError("Quaternion array must have shape (4,)")
        return Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
            frame=frame,
        )


@dataclass
class Translation:
    """Translations should be expressed in the to_ frame, which are typically the asset frame"""

    x: float
    y: float
    from_: Literal["robot", "asset"]
    to_: Literal["robot", "asset"]
    frame: Literal["asset", "robot"] = "asset"
    z: float = 0

    def as_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(
        coordinate: np.ndarray,
        from_: Literal["robot", "asset"],
        to_: Literal["robot", "asset"],
    ) -> Translation:
        if coordinate.shape != (3,):
            raise ValueError("Point should have shape (3,)")
        return Translation(
            x=coordinate[0], y=coordinate[1], z=coordinate[2], from_=from_, to_=to_
        )
