from typing import Literal, Union
import numpy as np
from alitra.frame_dataclasses import Euler, PointList, Point, Transform, Translation


class FrameTransform:
    """Let (from_) be the fixed local coordinate frame that the robot operate in, and (to_) be
     in the asset-fixed global coordinate system. Further, let the relationship between the two
     reference systems be described by : p_to = rot_transform(euler_angles)*p_from + translation,
      using the zyx Euler angle rotation convention. This object allows for transform between the frames
      (from_) and (to_). Can be rewritten to accommodate for more frames.
    :param euler: Euler angles transform (zyx convention).
    :param translation: translation from frame (from_) to frame (to_), expressed in the (to_) frame.
    """

    def __init__(
        self,
        euler: Euler,
        translation: Translation,
        from_: Literal["robot", "asset"],
        to_: Literal["robot", "asset"],
    ):
        try:
            self.transform = Transform(
                translation=translation, euler=euler, from_=from_, to_=to_
            )
        except ValueError as e:
            raise ValueError(e)

    def transform_point(
        self, coordinates: Union[Point, PointList], from_, to_
    ) -> Union[Point, PointList]:
        """Transforms a point or list of points from _from to _to (rotation and translation)"""
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
