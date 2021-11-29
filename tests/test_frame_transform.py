import math

import numpy as np
import pytest

from alitra.frame_dataclasses import (
    Euler,
    Point,
    PointList,
    Quaternion,
    Translation,
    Transform,
)
from alitra.frame_transform import FrameTransform


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_expected",
    [
        (
            Euler(psi=0.0, from_="robot", to_="asset"),
            Translation(x=0, y=0, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [1, 2, 3],
                        [-1, -2, -3],
                        [0, 0, 0],
                        [100000, 1, -100000],
                    ]
                ),
                frame="asset",
            ),
        ),
        (
            Euler(psi=np.pi * -0.0, from_="robot", to_="asset"),
            Translation(x=10, y=10, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [11, 12, 3],
                        [9, 8, -3],
                        [10, 10, 0],
                        [100010, 11, -100000],
                    ]
                ),
                frame="asset",
            ),
        ),
        (
            Euler(psi=np.pi / 2, from_="robot", to_="asset"),
            Translation(x=10, y=0, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [8, 1, 3],
                        [12, -1, -3],
                        [10, 0, 0],
                        [9, 100000, -100000],
                    ]
                ),
                frame="asset",
            ),
        ),
        (
            Euler(theta=1 * 0.2, phi=1, psi=0.4, from_="robot", to_="asset"),
            Translation(x=0, y=10, z=2, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [2.06950653e00, 9.30742421e00, 5.03932254e00],
                        [-2.06950653e00, 1.06925758e01, -1.03932254e00],
                        [0.00000000e00, 1.00000000e01, 2.00000000e00],
                        [4.76148230e04, 1.11500688e05, -7.28173316e04],
                    ]
                ),
                frame="asset",
            ),
        ),
    ],
)
def test_transform_list_of_points(eul_rot, ref_translations, p_expected):
    p_robot = PointList.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame="robot",
    )
    transform = Transform.from_euler_ZYX(
        euler=eul_rot,
        translation=ref_translations,
        from_=eul_rot.from_,
        to_=eul_rot.to_,
    )
    frame_transform = FrameTransform(transform=transform)
    p_asset = frame_transform.transform_point(p_robot, from_="robot", to_="asset")

    assert p_asset.frame == p_expected.frame
    assert np.allclose(p_expected.as_np_array(), p_asset.as_np_array())


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_expected",
    [
        (
            Euler(psi=math.pi / 2.0, from_="robot", to_="asset"),
            Translation(x=1, y=2, from_="robot", to_="asset"),
            Point.from_array(np.array([-1, 3, 3]), frame="asset"),
        ),
    ],
)
def test_transform_point(eul_rot, ref_translations, p_expected):

    p_robot = Point.from_array(np.array([1, 2, 3]), frame="robot")
    transform = Transform.from_euler_ZYX(
        euler=eul_rot,
        translation=ref_translations,
        from_=eul_rot.from_,
        to_=eul_rot.to_,
    )
    frame_transform = FrameTransform(transform)
    p_asset = frame_transform.transform_point(p_robot, from_="robot", to_="asset")

    assert p_asset.frame == p_expected.frame
    assert np.allclose(p_expected.as_np_array(), p_asset.as_np_array())


@pytest.mark.parametrize(
    "from_, to_, error_expected",
    [
        ("asset", "asset", True),
        ("robot", "robot", True),
        ("robot", "asset", False),
        ("asset", "robot", True),
    ],
)
def test_transform_point_error(from_, to_, error_expected):
    p_robot = PointList.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame="robot",
    )
    eul_rot = Euler(psi=0.0, from_="robot", to_="asset")
    translation = Translation(x=0, y=0, from_="robot", to_="asset")
    transform = Transform.from_euler_ZYX(
        euler=eul_rot, translation=translation, from_=eul_rot.from_, to_=eul_rot.to_
    )

    frame_transform = FrameTransform(transform)

    if error_expected:
        with pytest.raises(ValueError):
            frame_transform.transform_point(p_robot, from_=from_, to_=to_)
    else:
        frame_transform.transform_point(p_robot, from_=from_, to_=to_)


@pytest.mark.parametrize(
    "quaternion, rotation_quaternion, expected",
    [
        (
            Quaternion(x=0, y=0, z=0, w=1.0, frame="robot"),
            Quaternion(x=0, y=0, z=1, w=0, frame="robot"),
            Quaternion(x=0, y=0, z=1, w=0, frame="asset"),
        ),
        (
            Quaternion(x=0, y=0, z=0, w=1.0, frame="asset"),
            Quaternion(x=0, y=0, z=1, w=0, frame="robot"),
            Quaternion(x=0, y=0, z=1, w=0, frame="robot"),
        ),
    ],
)
def test_transform_quaternion(quaternion, rotation_quaternion, expected):
    translation: Translation = Translation(
        x=0, y=0, from_=quaternion.frame, to_=expected.frame, frame=expected.frame
    )
    transform = Transform.from_quat(
        quat=rotation_quaternion,
        translation=translation,
        from_=quaternion.frame,
        to_=expected.frame,
    )

    frame_transform: FrameTransform = FrameTransform(transform)

    rotated_quaternion: Quaternion = frame_transform.transform_quaternion(
        quaternion=quaternion, from_=quaternion.frame, to_=expected.frame
    )

    assert rotated_quaternion == expected
