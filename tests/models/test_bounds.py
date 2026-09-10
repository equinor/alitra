import numpy as np
import pytest

from alitra import Bounds, Position


def test_eq_bounds(default_bounds, robot_frame):
    expected_bounds = Bounds(
        Position(0, 0, 0, robot_frame), Position(1, 1, 1, robot_frame)
    )
    assert default_bounds == expected_bounds


def test_position_within_bounds(default_bounds, robot_frame):
    pos = Position(0.5, 0.5, 0.5, robot_frame)
    assert default_bounds.position_within_bounds(pos) == True


def test_position_outside_bounds(default_bounds, robot_frame):
    pos = Position(11, 11, 11, robot_frame)
    assert default_bounds.position_within_bounds(pos) == False


def test_position_wrong_frame(default_bounds, asset_frame):
    pos = Position(0.5, 0.5, 0.5, asset_frame)
    with pytest.raises(ValueError):
        default_bounds.position_within_bounds(pos)


@pytest.mark.parametrize(
    "position, expected_distance",
    [
        ((5, 25, 45), 0),  # Inside the bounds
        ((0, 25, 45), 0),  # On the surface of the bounds
        ((10, 25, 45), 0),  # On the opposite surface of the bounds
        ((-3, 25, 45), 3),  # Outside one face
        ((13, 25, 45), 3),  # Outside the opposite face
        ((-3, 16, 45), 5),  # Outside two faces, 3-4-5 triangle
        ((-3, 16, 62), 13),  # Outside three faces, 3-4-12 diagonal
    ],
)
def test_distance_to_position(robot_frame, position, expected_distance):
    bounds = Bounds(Position(0, 20, 40, robot_frame), Position(10, 30, 50, robot_frame))
    pos = Position(*position, robot_frame)

    assert np.allclose(bounds.distance_to_position(pos), expected_distance)


def test_distance_to_position_wrong_frame(default_bounds, asset_frame):
    pos = Position(0.5, 0.5, 0.5, asset_frame) # The default_bounds is in robot_frame
    with pytest.raises(ValueError):
        default_bounds.distance_to_position(pos)
