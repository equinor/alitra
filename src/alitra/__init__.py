"""
Alitra makes alignment and transformation between coordinate-frames
easier by using custom dataclasses

Imagine we have an "asset" with one coordinate-frame and a "robot" with its own
internal coordinate-frame. We want to transform a position in the robot coordinate-frame
to a position in the asset coordinate-frame.

>>> import numpy as np
>>> from alitra.models import Positions, Transform, Translation

Setting up rotations, translations and positions for transformation

>>> robot_frame = Frame("robot")
>>> asset_frame = Frame("asset")
>>> euler = np.array([np.pi / 4, 0, 0])
>>> translation = Translation(x=1, y=0, from_=robot_frame, to_=asset_frame)
>>> p_robot = Positions.from_array(np.array([[1, 1, 0], [10, 1, 0]]), frame=robot_frame)
>>> rotation_axes = "z"

Making the transform

>>> transform = Transform.from_euler_array(
...    translation=translation, euler=euler, from_=robot_frame, to_=asset_frame
... )

Tranform position on robot to a position on the asset

>>> p_asset = transform.transform_position(p_robot, from_=robot_frame, to_=asset_frame)

If you have one position in two different frames and the rotation between the axes you can find the transform
between the two frames.

>>> transform = Transform(p_robot, p_asset, rotation_axes)
"""

from alitra.alignment import align_maps, align_positions
from alitra.models import (
    Bounds,
    Frame,
    Map,
    MapAlignment,
    Orientation,
    Pose,
    Position,
    Positions,
    Translation,
)
from alitra.transform import Transform
