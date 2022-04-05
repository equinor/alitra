import numpy as np
import pytest

from alitra import Position, Positions


def test_position_array(robot_frame):
    expected_array = np.array([1, 1, 1])
    position: Position = Position.from_array(expected_array, frame=robot_frame)
    assert np.allclose(expected_array, position.to_array())


def test_position_invalid_array(robot_frame):
    with pytest.raises(ValueError):
        Position.from_array(np.array([1, 1]), frame=robot_frame)


def test_position_list_array(robot_frame):
    expected_array = np.array([[1, 1, 1], [2, 2, 2]])
    position_list: Positions = Positions.from_array(expected_array, frame=robot_frame)
    assert np.allclose(expected_array, position_list.to_array())


def test_positions_invalid_array(robot_frame):
    with pytest.raises(ValueError):
        Positions.from_array(np.array([[1, 1], [1, 1]]), frame=robot_frame)
