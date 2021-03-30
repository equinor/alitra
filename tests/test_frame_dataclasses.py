import numpy as np
import pytest

from alitra import (
    Euler,
    Point,
    PointList,
    Transform,
    Translation,
    Quaternion,
)


def test_translation():
    with pytest.raises(ValueError):
        Translation.from_array(np.array([1, 1]), from_="robot", to_="asset")


def test_euler():
    with pytest.raises(ValueError):
        Euler.from_array(np.array([1, 1]), from_="robot", to_="asset")


def test_quaternion():
    with pytest.raises(ValueError):
        Quaternion.from_array(np.array([1, 2, 3]), frame="asset")


def test_point():
    with pytest.raises(ValueError):
        Point.from_array(np.array([1, 1]), frame="robot")


def test_point_list():
    with pytest.raises(ValueError):
        PointList.from_array(np.array([[1, 1], [1, 1]]), frame="robot")


def test_transform():
    with pytest.raises(ValueError):
        euler = Euler(from_="robot", to_="asset")
        translation = Translation(1, 1, from_="robot", to_="asset")
        Transform.from_euler_ZYX(translation, euler=euler, from_="asset", to_="robot")


def test_transform_without_rotation():
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_="robot", to_="asset")
        Transform(translation=translation, from_="asset", to_="robot")
