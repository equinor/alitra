from typing import Literal, Union

import numpy as np
from scipy.spatial.transform.rotation import Rotation

from alitra.frame_dataclasses import Point, PointList, Quaternion, Transform


class FrameTransform:
    """Let (from_) be the fixed local coordinate frame that the robot operate in, and (to_) be
    in the asset-fixed global coordinate system. Further, let the relationship between the two
    reference systems be described by : p_to = rotation_object.apply(p_from) + translation.
    This object allows for transform between the frames (from_) and (to_). Can be rewritten to
    accommodate for multiple frames.
    :param transform: A transform dataclass object
    """

    def __init__(self, transform: Transform):
        self.transform = transform

    def transform_point(
        self,
        coordinates: Union[Point, PointList],
        from_: Literal["robot", "asset"],
        to_: Literal["robot", "asset"],
    ) -> Union[Point, PointList]:
        """
        Transforms a point or list of points from _from to _to (rotation and translation)
        :param coordinates: Point or PointList of coordinates to transform.
        :param from_: Source coordinate system. Must be "robot" or "asset" and different to "to_".
        :param to_: Destination coordinate system. Must be "robot" or "asset" and different to "from_".
        :return: Point or PointList with coordinates in the to_ coordinate system.
        """
        if from_ == to_:
            return coordinates

        if coordinates.frame != from_:
            raise ValueError(
                f"Expected coordinates in frame {from_} "
                + f", got coordinates in frame {coordinates.frame}"
            )

        result: np.ndarray
        if from_ == self.transform.to_ and to_ == self.transform.from_:
            """Using the inverse transform"""
            result = self.transform.rotation_object.apply(
                coordinates.as_np_array() - self.transform.translation.as_np_array(),
                inverse=True,
            )

        elif from_ == self.transform.from_ and to_ == self.transform.to_:
            result = (
                self.transform.rotation_object.apply(coordinates.as_np_array())
                + self.transform.translation.as_np_array()
            )
        else:
            raise ValueError("Transform not specified")

        if isinstance(coordinates, Point):
            return Point.from_array(result, to_)
        elif isinstance(coordinates, PointList):
            return PointList.from_array(result, to_)
        else:
            raise ValueError("Incorrect input format. Must be Point or PointList.")

    def transform_quaternion(
        self,
        quaternion: Quaternion,
        from_: Literal["robot", "asset"],
        to_: Literal["robot", "asset"],
    ) -> Quaternion:
        if from_ == to_:
            return quaternion

        quaternion_from: Rotation = Rotation.from_quat(quaternion.as_np_array())

        if not isinstance(quaternion, Quaternion):
            raise ValueError("Incorrect input format. Must be Quaternion.")

        if from_ == self.transform.to_ and to_ == self.transform.from_:
            """Using the inverse transform"""

            quaternion_to = quaternion_from * self.transform.rotation_object.inv()
        elif from_ == self.transform.from_ and to_ == self.transform.to_:
            quaternion_to = quaternion_from * self.transform.rotation_object
        else:
            raise ValueError("Transform not specified")

        result: Quaternion = Quaternion.from_array(quaternion_to.as_quat(), frame=to_)
        return result
