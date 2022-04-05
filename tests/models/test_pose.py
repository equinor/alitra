import numpy as np

from alitra import Frame, Pose


def test_pose_array():
    expected_pos_array = np.array([0, 0, 0])
    expected_quat_array = np.array([0, 0, 0, 1])
    expected_frame = Frame("robot")

    pose: Pose = Pose.from_array(
        expected_pos_array, expected_quat_array, expected_frame
    )
    pos_array, quat_array = pose.to_array()
    assert pose.frame == expected_frame
    assert np.allclose(pos_array, expected_pos_array)
    assert np.allclose(quat_array, expected_quat_array)
