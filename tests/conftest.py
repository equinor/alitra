from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alitra import (
    Bounds,
    Frame,
    Map,
    Orientation,
    Pose,
    Position,
    Positions,
    Transform,
    Translation,
)


@pytest.fixture()
def robot_frame():
    return Frame("robot")


@pytest.fixture()
def asset_frame():
    return Frame("asset")


@pytest.fixture()
def default_position(robot_frame):
    return Position(x=0, y=0, z=0, frame=robot_frame)


@pytest.fixture()
def robot_position_1(robot_frame):
    return Position(x=0, y=0, z=0, frame=robot_frame)


@pytest.fixture()
def robot_position_2(robot_frame):
    return Position(x=1, y=1, z=1, frame=robot_frame)


@pytest.fixture()
def robot_positions(robot_position_1, robot_position_2, robot_frame):
    return Positions([robot_position_1, robot_position_2], frame=robot_frame)


@pytest.fixture()
def default_rotation():
    return Rotation.from_quat(np.array([0, 0, 0, 1]))


@pytest.fixture()
def default_orientation(robot_frame):
    return Orientation.from_quat_array(np.array([0, 0, 0, 1]), frame=robot_frame)


@pytest.fixture()
def default_pose(default_position, default_orientation, robot_frame):
    return Pose(
        position=default_position, orientation=default_orientation, frame=robot_frame
    )


@pytest.fixture()
def default_translation(robot_frame, asset_frame):
    return Translation(x=0, y=0, z=0, from_=robot_frame, to_=asset_frame)


@pytest.fixture()
def default_transform(default_translation, default_rotation, robot_frame, asset_frame):
    return Transform(
        translation=default_translation,
        rotation=default_rotation,
        from_=robot_frame,
        to_=asset_frame,
    )


@pytest.fixture()
def default_bounds(robot_position_1, robot_position_2):
    return Bounds(robot_position_1, robot_position_2)


@pytest.fixture()
def robot_map():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_robot.json"))
    return Map.from_config(map_path)


@pytest.fixture()
def asset_map():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_asset.json"))
    return Map.from_config(map_path)


@pytest.fixture()
def map_with_bounds():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_bounds.json"))
    return Map.from_config(map_path)
