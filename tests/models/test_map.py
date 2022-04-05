from pathlib import Path

import pytest

from alitra import Bounds, Frame, Map, MapAlignment, Position, Positions

robot_frame = Frame("robot")

expected_map = Map(
    name="test_map_robot",
    reference_positions=Positions(
        positions=[
            Position(x=20, y=10, z=0, frame=robot_frame),
            Position(x=60, y=20, z=0, frame=robot_frame),
            Position(x=80, y=70, z=0, frame=robot_frame),
        ],
        frame=robot_frame,
    ),
    frame=robot_frame,
)

expected_map_bounds = Map(
    name="test_map_bounds",
    reference_positions=Positions(
        positions=[
            Position(x=20, y=10, z=0, frame=robot_frame),
            Position(x=60, y=20, z=0, frame=robot_frame),
            Position(x=80, y=70, z=0, frame=robot_frame),
        ],
        frame=robot_frame,
    ),
    frame=robot_frame,
    bounds=Bounds(
        position1=Position(x=0, y=0, z=0, frame=robot_frame),
        position2=Position(x=100, y=100, z=100, frame=robot_frame),
    ),
)


def test_load_map():
    map_path = Path("./tests/test_data/test_map_robot.json")
    map: Map = Map.from_config(map_path)
    assert map == expected_map


def test_invalid_file_path():
    map_path = Path("./tests/test_data/no_file.json")
    with pytest.raises(Exception):
        Map.from_config(map_path)


def test_load_map_bounds():
    map_path = Path("./tests/test_data/test_map_bounds.json")
    map: Map = Map.from_config(map_path)
    assert map == expected_map_bounds
    assert map.bounds.x_min == expected_map_bounds.bounds.position1.x
    assert map.bounds.x_min == expected_map_bounds.bounds.position1.x
    assert map.bounds.y_min == expected_map_bounds.bounds.position1.y
    assert map.bounds.y_max == expected_map_bounds.bounds.position2.y
    assert map.bounds.z_max == expected_map_bounds.bounds.position2.z
    assert map.bounds.z_max == expected_map_bounds.bounds.position2.z


def test_mapalignment():
    map_path = Path("./tests/test_data/test_mapalignment.json")
    map_alignment: MapAlignment = MapAlignment.from_config(map_path)
    assert map_alignment.map_from == expected_map
