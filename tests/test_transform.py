import pytest

from alitra import Transform, Translation


def test_transform(default_rotation, robot_frame, asset_frame):
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform(
            translation, rotation=default_rotation, from_=asset_frame, to_=robot_frame
        )


def test_transform_without_rotation(robot_frame, asset_frame):
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform(translation=translation, from_=asset_frame, to_=robot_frame)


import numpy as np
import pytest

from alitra import Frame, Orientation, Pose, Position, Positions, Transform, Translation


@pytest.mark.parametrize(
    "euler_array, translation, expected_position",
    [
        (
            np.array([0, 0, 0]),
            Translation(x=0, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [1, 2, 3],
                        [-1, -2, -3],
                        [0, 0, 0],
                        [100000, 1, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            np.array([np.pi * -0.0, 0, 0]),
            Translation(x=10, y=10, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [11, 12, 3],
                        [9, 8, -3],
                        [10, 10, 0],
                        [100010, 11, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            np.array([np.pi / 2, 0, 0]),
            Translation(x=10, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [8, 1, 3],
                        [12, -1, -3],
                        [10, 0, 0],
                        [9, 100000, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            np.array([0.4, 0.2, 1]),
            Translation(x=0, y=10, z=2, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [2.06950653e00, 9.30742421e00, 5.03932254e00],
                        [-2.06950653e00, 1.06925758e01, -1.03932254e00],
                        [0.00000000e00, 1.00000000e01, 2.00000000e00],
                        [4.76148230e04, 1.11500688e05, -7.28173316e04],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
    ],
)
def test_transform_list_of_positions(euler_array, translation, expected_position):
    p_robot = Positions.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame=Frame("robot"),
    )
    transform = Transform.from_euler_array(
        euler=euler_array,
        translation=translation,
        from_=translation.from_,
        to_=translation.to_,
    )

    p_asset = transform.transform_position(
        p_robot, from_=Frame("robot"), to_=Frame("asset")
    )

    assert p_asset.frame == expected_position.frame
    assert np.allclose(expected_position.to_array(), p_asset.to_array())


def test_no_transformation_when_equal_frames():
    p_expected = Position.from_array(np.array([1, 2, 3]), frame=Frame("robot"))

    translation = Translation(x=2, y=3, from_=Frame("robot"), to_=Frame("asset"))
    transform = Transform.from_euler_array(
        euler=np.array([1.0, 0, 0]),
        translation=translation,
        from_=translation.from_,
        to_=translation.to_,
    )

    p_robot = Position.from_array(np.array([1, 2, 3]), frame=Frame("robot"))
    position = transform.transform_position(
        p_robot, from_=Frame("robot"), to_=Frame("robot")
    )

    assert position.frame == p_expected.frame
    assert np.allclose(p_expected.to_array(), position.to_array())


@pytest.mark.parametrize(
    "from_, to_, error_expected",
    [
        (Frame("asset"), Frame("asset"), True),
        (Frame("robot"), Frame("robot"), False),
        (Frame("robot"), Frame("asset"), False),
        (Frame("asset"), Frame("robot"), True),
    ],
)
def test_transform_position_error(from_, to_, error_expected):
    p_robot = Positions.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame=Frame("robot"),
    )
    euler = np.array([0, 0, 0])
    translation = Translation(x=0, y=0, from_=Frame("robot"), to_=Frame("asset"))
    transform = Transform.from_euler_array(
        euler=euler, translation=translation, from_=Frame("robot"), to_=Frame("asset")
    )

    if error_expected:
        with pytest.raises(ValueError):
            transform.transform_position(p_robot, from_=from_, to_=to_)
    else:
        transform.transform_position(p_robot, from_=from_, to_=to_)


def test_transform_position(default_transform, robot_frame, asset_frame):
    expected_pos = Position(1, 1, 1, asset_frame)
    pos = Position(1, 1, 1, robot_frame)
    pos_to = default_transform.transform_position(
        positions=pos, from_=robot_frame, to_=asset_frame
    )
    assert np.allclose(expected_pos.to_array(), pos_to.to_array())


def test_transform_orientation(
    default_transform, default_orientation, robot_frame, asset_frame
):
    expected_orientation: Orientation = Orientation.from_quat_array(
        np.array([0, 0, 0, 1]), frame=asset_frame
    )

    orientation: Orientation = default_transform.transform_orientation(
        orientation=default_orientation,
        from_=robot_frame,
        to_=asset_frame,
    )
    assert np.allclose(
        expected_orientation.to_quat_array(), orientation.to_quat_array()
    )


def test_transform_pose(
    default_transform, default_pose, default_rotation, robot_frame, asset_frame
):
    expected_pose: Pose = Pose(
        position=Position(x=0, y=0, z=0, frame=asset_frame),
        orientation=Orientation.from_rotation(
            rotation=default_rotation, frame=asset_frame
        ),
        frame=asset_frame,
    )

    pose_to: Pose = default_transform.transform_pose(
        pose=default_pose,
        from_=robot_frame,
        to_=asset_frame,
    )
    assert np.allclose(
        expected_pose.orientation.to_quat_array(), pose_to.orientation.to_quat_array()
    )
    assert np.allclose(expected_pose.position.to_array(), pose_to.position.to_array())
    assert expected_pose.frame == pose_to.frame
