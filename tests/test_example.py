from pathlib import Path

import numpy as np

from alitra import Frame, Map, Orientation, Pose, Position, Transform, align_maps


def test_example():
    asset_frame = Frame("asset")
    expected_pose: Pose = Pose(
        Position(x=40, y=40, z=0, frame=asset_frame),
        Orientation.from_euler_array(np.array([-np.pi / 2, 0, 0]), frame=asset_frame),
        frame=asset_frame,
    )
    """
    This test is an example of one way to use alitra for transformations and alignment
    between two coordinate frames

    Assume we have two maps, './test_data/test_map_config.json' and
    './test_data/test_map_config_asset.json' which represents a map created by a
    robot and a map of an asset respectively. First we load the maps as models
    """

    here = Path(__file__).parent.resolve()

    robot_map_path = Path(here.joinpath("./test_data/test_map_robot.json"))
    robot_map: Map = Map.from_config(robot_map_path)

    asset_map_path = Path(here.joinpath("./test_data/test_map_asset.json"))
    asset_map: Map = Map.from_config(asset_map_path)

    """
    Now we create the transform between the two maps, we know that the only difference
    between the two maps are a rotation about the z-axis
    """

    transform: Transform = align_maps(robot_map, asset_map, rot_axes="z")

    """
    We now create a Pose in the robot frame where the robot is standing
    """

    position: Position = Position(x=30, y=40, z=0, frame=robot_map.frame)
    orientation: Orientation = Orientation.from_euler_array(
        np.array([np.pi, 0, 0]), frame=robot_map.frame
    )
    robot_pose: Pose = Pose(
        position=position, orientation=orientation, frame=robot_map.frame
    )

    """
    Now we can transform the robot_pose into the asset_map to know where on our asset
    the robot is
    """

    asset_pose = transform.transform_pose(
        pose=robot_pose,
        from_=robot_pose.frame,
        to_=asset_map.frame,
    )

    assert np.allclose(
        expected_pose.orientation.to_euler_array(),
        asset_pose.orientation.to_euler_array(),
    )
    assert np.allclose(
        expected_pose.position.to_array(), asset_pose.position.to_array()
    )
