from cmath import exp
from pathlib import Path

import pytest

from alitra.frame_dataclasses import Point, PointList
from alitra.models.bounds import Bounds, FrameException
from alitra.models.map_config import MapConfig, load_map_config

expected_map_config = MapConfig(
    map_name="test_map",
    robot_reference_points=PointList(
        points=[
            Point(x=10, y=20, z=30, frame="robot"),
            Point(x=40, y=50, z=60, frame="robot"),
            Point(x=70, y=80, z=90, frame="robot"),
        ],
        frame="robot",
    ),
    asset_reference_points=PointList(
        points=[
            Point(x=11, y=21, z=31, frame="asset"),
            Point(x=41, y=51, z=61, frame="asset"),
            Point(x=71, y=81, z=91, frame="asset"),
        ],
        frame="asset",
    ),
)
expected_map_config_bounds = MapConfig(
    map_name="test_map",
    robot_reference_points=PointList(
        points=[
            Point(x=10, y=20, z=30, frame="robot"),
            Point(x=40, y=50, z=60, frame="robot"),
            Point(x=70, y=80, z=90, frame="robot"),
        ],
        frame="robot",
    ),
    asset_reference_points=PointList(
        points=[
            Point(x=11, y=21, z=31, frame="asset"),
            Point(x=41, y=51, z=61, frame="asset"),
            Point(x=71, y=81, z=91, frame="asset"),
        ],
        frame="asset",
    ),
    bounds=Bounds(
        point1=Point(x=5, y=15, z=30, frame="asset"),
        point2=Point(x=80, y=90, z=100, frame="asset"),
    ),
)


def test_load_map_config():
    map_config_path = Path("./tests/test_data/test_map_config/test_map_config.json")
    map_config: MapConfig = load_map_config(map_config_path)
    assert map_config == expected_map_config


def test_invalid_file_path():
    map_config_path = Path("./tests/test_data/test_map_config/no_file.json")
    with pytest.raises(Exception):
        load_map_config(map_config_path)


def test_load_map_bounds():
    map_config_path = Path(
        "./tests/test_data/test_map_config/test_map_config_bounds.json"
    )
    map_config: MapConfig = load_map_config(map_config_path)
    assert map_config == expected_map_config_bounds
    assert map_config.bounds.x_min == expected_map_config_bounds.bounds.point1.x
    assert map_config.bounds.x_min == expected_map_config_bounds.bounds.point1.x
    assert map_config.bounds.y_min == expected_map_config_bounds.bounds.point1.y
    assert map_config.bounds.y_max == expected_map_config_bounds.bounds.point2.y
    assert map_config.bounds.z_max == expected_map_config_bounds.bounds.point2.z
    assert map_config.bounds.z_max == expected_map_config_bounds.bounds.point2.z


def test_points_within_bounds():
    map_config_path = Path(
        "./tests/test_data/test_map_config/test_map_config_bounds.json"
    )
    map_config: MapConfig = load_map_config(map_config_path)
    point_within_bounds: Point = Point(x=10, y=40, z=50, frame="asset")
    point_outside_bounds: Point = Point(x=2, y=40, z=50, frame="asset")
    point_with_wrong_frame: Point = Point(x=2, y=40, z=50, frame="robot")

    assert map_config.bounds.point_within_bounds(point=point_within_bounds) == True
    assert map_config.bounds.point_within_bounds(point=point_outside_bounds) == False
    with pytest.raises(FrameException):
        map_config.bounds.point_within_bounds(point=point_with_wrong_frame)
