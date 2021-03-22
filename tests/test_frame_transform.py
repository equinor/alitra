import numpy as np
import pytest

from alitra import Euler, FrameTransform, Point, PointList, Translation


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
            Euler(phi=1 * 0.2, theta=1, psi=0.4, from_="robot", to_="asset"),
            Translation(x=0, y=10, z=2, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [0.73539728, 8.75538989, 5.45110656],
                        [-0.73539728, 11.24461010, -1.45110656],
                        [0, 10, 2],
                        [70402.7949, 118918.343, -30068.7893],
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

    frame_transform = FrameTransform(
        eul_rot, ref_translations, from_=eul_rot.from_, to_=eul_rot.to_
    )
    p_asset = frame_transform.transform_point(p_robot, from_="robot", to_="asset")

    assert p_asset.frame == p_expected.frame
    assert np.allclose(p_expected.as_np_array(), p_asset.as_np_array())


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_expected",
    [
        (
            Euler(psi=0.0, from_="robot", to_="asset"),
            Translation(x=0, y=0, from_="robot", to_="asset"),
            Point.from_array(np.array([1, 2, 3]), frame="asset"),
        ),
    ],
)
def test_transform_point(eul_rot, ref_translations, p_expected):

    p_robot = Point.from_array(np.array([1, 2, 3]), frame="robot")

    frame_transform = FrameTransform(
        eul_rot, ref_translations, from_=eul_rot.from_, to_=eul_rot.to_
    )
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

    frame_transform = FrameTransform(
        eul_rot, translation, from_=eul_rot.from_, to_=eul_rot.to_
    )

    if error_expected:
        with pytest.raises(ValueError):
            frame_transform.transform_point(p_robot, from_=from_, to_=to_)
    else:
        frame_transform.transform_point(p_robot, from_=from_, to_=to_)
