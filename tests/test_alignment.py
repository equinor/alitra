import numpy as np
import pytest

from alitra import Frame, Position, Positions, Transform, align_maps, align_positions


def test_align_positions_translation_only():
    expected_rotation = np.array([0, 0, 0])
    expected_translation = np.array([1, -1, 0])

    robot_frame = Frame("robot")
    robot_positions: Positions = Positions.from_array(
        np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 2, 0],
            ]
        ),
        frame=robot_frame,
    )

    asset_frame = Frame("asset")
    asset_positions: Positions = Positions.from_array(
        np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [1, 1, 0],
            ]
        ),
        frame=asset_frame,
    )

    transform: Transform = align_positions(
        positions_from=robot_positions, positions_to=asset_positions, rot_axes="xyz"
    )

    assert np.allclose(expected_rotation, transform.rotation.as_euler("ZYX"))
    assert np.allclose(expected_translation, transform.translation.to_array())


def test_align_positions_one_dimension():

    expected_rotation = np.array([np.pi / 2, 0, 0])
    expected_translation = np.array([0, 0, 0])

    robot_frame = Frame("robot")
    robot_positions: Positions = Positions.from_array(
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 0],
            ]
        ),
        frame=robot_frame,
    )

    asset_frame = Frame("asset")
    asset_positions: Positions = Positions.from_array(
        np.array(
            [
                [-1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        ),
        frame=asset_frame,
    )

    transform: Transform = align_positions(
        positions_from=robot_positions, positions_to=asset_positions, rot_axes="xyz"
    )

    assert np.allclose(expected_rotation, transform.rotation.as_euler("ZYX"))
    assert np.allclose(expected_translation, transform.translation.to_array())


def test_align_positions_three_dimensions():

    expected_rotation = np.array([np.pi / 2, np.pi / 4, np.pi / 2])
    expected_translation = np.array([0, 0, 0])
    hyp = np.sqrt(1 / 2)

    robot_frame = Frame("robot")
    robot_positions: Positions = Positions.from_array(
        np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ]
        ),
        frame=robot_frame,
    )

    asset_frame = Frame("asset")
    asset_positions: Positions = Positions.from_array(
        np.array(
            [
                [0, 0, 1],
                [-hyp, -hyp, 1],
                [-hyp, -hyp, 0],
            ]
        ),
        frame=asset_frame,
    )

    transform: Transform = align_positions(
        positions_from=robot_positions, positions_to=asset_positions, rot_axes="xyz"
    )

    assert np.allclose(expected_rotation, transform.rotation.as_euler("zyx"))
    assert np.allclose(expected_translation, transform.translation.to_array())


def test_align_positions_rotation_and_translation():

    expected_rotation = np.array([np.pi / 2, 0, 0])
    expected_translation = np.array([2, -1, 0])

    robot_frame = Frame("robot")
    robot_positions: Positions = Positions.from_array(
        np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [1, 1, 0],
            ]
        ),
        frame=robot_frame,
    )

    asset_frame = Frame("asset")
    asset_positions: Positions = Positions.from_array(
        np.array(
            [
                [2, 0, 0],
                [2, 1, 0],
                [1, 0, 0],
            ]
        ),
        frame=asset_frame,
    )

    transform: Transform = align_positions(
        positions_from=robot_positions, positions_to=asset_positions, rot_axes="xyz"
    )

    assert np.allclose(expected_rotation, transform.rotation.as_euler("ZYX"))
    assert np.allclose(expected_translation, transform.translation.to_array())


def test_align_maps(robot_map, asset_map, default_position, robot_frame, asset_frame):
    expected_position: Position = Position(80, 10, 0, frame=asset_frame)

    transform = align_maps(map_from=robot_map, map_to=asset_map, rot_axes="z")

    position_to = transform.transform_position(
        positions=default_position,
        from_=robot_frame,
        to_=asset_frame,
    )
    assert np.allclose(expected_position.to_array(), position_to.to_array())
    assert expected_position.frame == position_to.frame


def test_align_positions_unequal_position_length(robot_frame, asset_frame):
    positions_from = Positions.from_array(
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]), frame=robot_frame
    )
    positions_to = Positions.from_array(
        np.array([[1, 0, 0], [1, 0, 0]]), frame=asset_frame
    )
    with pytest.raises(ValueError):
        align_positions(
            positions_from=positions_from,
            positions_to=positions_to,
            rot_axes="xyz",
        )


def test_align_positions_not_enough_positions_one_rotation(robot_frame, asset_frame):
    positions_from = Positions.from_array(np.array([[1, 0, 0]]), frame=robot_frame)
    positions_to = Positions.from_array(np.array([[1, 0, 0]]), frame=asset_frame)
    with pytest.raises(ValueError):
        align_positions(
            positions_from=positions_from, positions_to=positions_to, rot_axes="z"
        )


def test_align_positions_not_enough_positions_three_rotations(robot_frame, asset_frame):
    positions_from = Positions.from_array(
        np.array([[1, 0, 0], [1, 0, 0]]), frame=robot_frame
    )
    positions_to = Positions.from_array(
        np.array([[1, 0, 0], [1, 0, 0]]), frame=asset_frame
    )
    with pytest.raises(ValueError):
        align_positions(
            positions_from=positions_from, positions_to=positions_to, rot_axes="xyz"
        )


def test_align_positions_outside_rsme_treshold(robot_frame, asset_frame):
    positions_from = Positions.from_array(
        np.array([[20, 10, 0], [60, 20, 0], [80, 70, 0]]), frame=robot_frame
    )

    positions_to = Positions.from_array(
        np.array([[10, 20, 0], [30, 40, 0], [50, 60, 0]]), frame=asset_frame
    )
    with pytest.raises(ValueError):
        align_positions(
            positions_from=positions_from, positions_to=positions_to, rot_axes="z"
        )
