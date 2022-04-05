import numpy as np
import pytest

from alitra import Orientation


def test_orientation_quat_array(robot_frame):
    expected_array = np.array([0.5, 0.5, 0.5, 0.5])
    orientation: Orientation = Orientation.from_quat_array(expected_array, robot_frame)
    assert np.allclose(orientation.to_quat_array(), expected_array)


def test_orientation_euler_array(robot_frame):
    expected_euler = np.array([1, 1, 1])
    orientation: Orientation = Orientation.from_euler_array(expected_euler, robot_frame)
    assert np.allclose(orientation.to_euler_array(), expected_euler)


def test_orientation_invalid_quat_array(robot_frame):
    with pytest.raises(ValueError):
        Orientation.from_quat_array(np.array([1, 1]), frame=robot_frame)


def test_orientation_invalid_euler_array(robot_frame):
    with pytest.raises(ValueError):
        Orientation.from_euler_array(np.array([1, 1]), frame=robot_frame)
