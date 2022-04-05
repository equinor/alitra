from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from .models.frame import Frame
from .models.orientation import Orientation
from .models.pose import Pose
from .models.position import Position, Positions
from .models.translation import Translation


@dataclass
class Transform:
    """
    A transform object that describe the transformation between two frames.
    Contains a scipy rotation object, a translation and two frames.
    Can be created from euler array or quaternion array. Translations must be
    expressed in the (to_) frame
    """

    translation: Translation
    from_: Frame
    to_: Frame
    rotation: Rotation = None

    def __post_init__(self):
        if (
            not self.translation.from_ == self.from_
            and not self.translation.to_ == self.to_
        ):
            raise ValueError(
                f"The from_ frames or to_ frames of translation and transform object are not equal."
            )

    def transform_position(
        self,
        positions: Union[Position, Positions],
        from_: Frame,
        to_: Frame,
    ) -> Union[Position, Positions]:
        """
        Transforms a position or list of positions from from_ to to_ (rotation and translation)
        :param positions: Position or Positions in the from_ coordinate system.
        :param from_: Source Frame, must be different to "to_".
        :param to_: Destination Frame, must be different to "from_".
        :return: Position or Positions in the to_ coordinate system.
        """
        if positions.frame != from_:
            raise ValueError(
                f"Expected positions in frame {from_} "
                + f", got positions in frame {positions.frame}"
            )

        if from_ == to_:
            return positions

        result: np.ndarray
        if from_ == self.to_ and to_ == self.from_:
            """Using the inverse transform"""
            result = self.rotation.apply(
                positions.to_array() - self.translation.to_array(),
                inverse=True,
            )
        elif from_ == self.from_ and to_ == self.to_:
            result = (
                self.rotation.apply(positions.to_array()) + self.translation.to_array()
            )
        else:
            raise ValueError("Transform not specified")

        if isinstance(positions, Position):
            return Position.from_array(result, to_)
        elif isinstance(positions, Positions):
            return Positions.from_array(result, to_)
        else:
            raise ValueError("Incorrect input format. Must be Position or Positions.")

    def transform_rotation(
        self, rotation: Rotation, from_: Frame, to_: Frame
    ) -> Rotation:
        """
        Transforms a rotation from from_ to to_ (rotation)
        :param rotation: Rotation (scipy) in the from_ coordinate system.
        :param from_: Source Frame, must be different to "to_".
        :param to_: Destination Frame, must be different to "from_".
        :return: Rotation in the to_ coordinate system.
        """

        if from_ == self.to_ and to_ == self.from_:
            "Using the inverse transform"
            rotation_to = rotation * self.rotation.inv()
        elif from_ == self.from_ and to_ == self.to_:
            rotation_to = rotation * self.rotation
        else:
            raise ValueError("Transform not specified")

        return rotation_to

    def transform_orientation(
        self,
        orientation: Orientation,
        from_: Frame,
        to_: Frame,
    ) -> Orientation:
        """
        Transforms an orientation from from_ to to_ (rotation)
        :param orientation: Orientation in the from_ coordinate system.
        :param from_: Source Frame, must be different to "to_".
        :param to_: Destination Frame, must be different to "from_".
        :return: Orientation in the to_ coordinate system.
        """

        if from_ == to_:
            return orientation

        rotation_to = self.transform_rotation(orientation.to_rotation(), from_, to_)
        return Orientation(*rotation_to.as_quat(), frame=to_)  # type: ignore

    def transform_pose(self, pose: Pose, from_: Frame, to_: Frame) -> Pose:
        """
        Transforms a pose from from_ to to_ (rotation)
        :param pose: Pose in the from_ coordinate system.
        :param from_: Source Frame, must be different to "to_".
        :param to_: Destination Frame, must be different to "from_".
        :return: Pose in the to_ coordinate system.
        """

        if from_ == to_:
            return pose

        position = self.transform_position(pose.position, from_, to_)
        if not isinstance(position, Position):
            raise TypeError("Pose can only contain a single position, not positions")
        orientation: Orientation = self.transform_orientation(
            pose.orientation, from_, to_
        )

        return Pose(position, orientation, to_)

    @staticmethod
    def from_euler_array(
        translation: Translation, euler: np.ndarray, from_: Frame, to_: Frame, seq="ZYX"
    ) -> Transform:
        """
        :param translation: Translation object between two frames
        :param euler: Numpy array of euler angles [x,y,z], shape (3,)
        :param from_: Frame the transform is coming from
        :param to_: Frame the transform is going to
        :param seq: Sequence of axes for rotations, same as scipy rotation
        :return: Transform object
        """
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation=Rotation.from_euler(seq=seq, angles=euler),
        )

    @staticmethod
    def from_quat_array(
        translation: Translation,
        quat: np.ndarray,
        from_: Frame,
        to_: Frame,
    ) -> Transform:
        """
        :param translation: Translation object between two frames
        :param quat: Numpy array of quaternions [x,y,z,w], shape (4,)
        :param from_: Frame the transform is coming from
        :param to_: Frame the transform is going to
        :return: Transform object
        """
        rotation = Rotation.from_quat(quat)
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation=rotation,
        )
