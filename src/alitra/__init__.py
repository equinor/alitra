"""
Alitra makes alignment and transformation between coordinate-frames
easier by using custom dataclasses

Imagine we have an "asset" with one coordinate-frame and a "robot" with its own
internal coordinate-frame. We want to transform a point on the robot to a point in the
asset coordinate-frame.

>>> import numpy as np
>>> from alitra.align_frames import AlignFrames
>>> from alitra.frame_dataclasses import Euler, Quaternion, PointList, Translation
>>> from alitra.frame_transform import FrameTransform

Setting up rotations, translations and points for transformation

>>> eul_rot = Euler(psi=np.pi / 4, from_="robot", to_="asset").as_np_array()
>>> ref_translations = Translation(x=1, y=0, from_="robot", to_="asset")
>>> p_robot = PointList.from_array(np.array([[1, 1, 0], [10, 1, 0]]), frame="robot")
>>> rotation_axes = "z"

Making the transform

>>> c_frame_transform = FrameTransform(
...    eul_rot, ref_translations, from_=eul_rot.from_, to_=eul_rot.to_
... )

Tranform point on robot to a point on the asset

>>> p_asset = c_frame_transform.transform_point(p_robot, from_="robot", to_="asset")

If you have one point in two different frames and the rotation between the axes you can find the transform
between the two frames.

>>> transform = AlignFrames.align_frames(p_robot, p_asset, rotation_axes)

"""

from alitra.align_frames import AlignFrames
from alitra.frame_dataclasses import (
    Euler,
    Point,
    PointList,
    Quaternion,
    Transform,
    Translation,
)
from alitra.frame_transform import FrameTransform
from alitra.models.bounds import Bounds
from alitra.models.map_config import MapConfig, load_map_config
