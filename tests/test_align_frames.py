import numpy as np
import pytest

from alitra import AlignFrames, Euler, FrameTransform, PointList, Translation


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_robot,rotation_axes",
    [
        (
            Euler(psi=np.pi * -0.0, from_="robot", to_="asset"),
            Translation(x=200030, y=10000, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [10, 1, 0],
                        [20, 2, 0],
                        [30, 7, 0],
                        [40, 5, 0],
                    ]
                ),
                frame="robot",
            ),
            "z",
        ),
        (
            Euler(psi=np.pi / 2, from_="robot", to_="asset"),
            Translation(x=10, y=0, from_="robot", to_="asset"),
            PointList.from_array(
                np.array([[5, 0, 0], [5, 2, 0], [7, 5, 0], [3, 5, 0]]),
                frame="robot",
            ),
            "z",
        ),
        (
            Euler(theta=np.pi * 0.9, from_="robot", to_="asset"),
            Translation(x=1, y=10, from_="robot", to_="asset"),
            PointList.from_array(
                np.array([[10, 0, 0], [5, 2, 0], [7, 5, 0], [3, 5, 0]]), frame="robot"
            ),
            "x",
        ),
        (
            Euler(phi=1 * 0.2, theta=1, psi=0.4, from_="robot", to_="asset"),
            Translation(x=0, y=10, z=2, from_="robot", to_="asset"),
            PointList.from_array(
                np.array(
                    [
                        [0, 1, 2],
                        [5, 2, 6],
                        [7, 5, 0],
                        [3, 5, 0],
                        [3, 5, 10],
                        [3, 5, 11],
                    ]
                ),
                frame="robot",
            ),
            "xyz",
        ),
        (
            Euler(psi=np.pi / 4, from_="robot", to_="asset"),
            Translation(x=1, y=0, from_="robot", to_="asset"),
            PointList.from_array(np.array([[1, 1, 0], [10, 1, 0]]), frame="robot"),
            "z",
        ),
        (
            Euler(phi=np.pi * 0.2, from_="robot", to_="asset"),
            Translation(x=1, y=10, z=2, from_="robot", to_="asset"),
            PointList.from_array(
                np.array([[0, 1, 2], [5, 2, 0], [7, 5, 0], [3, 5, 0]]), frame="robot"
            ),
            "y",
        ),
    ],
)
def test_align_frames(eul_rot, ref_translations, p_robot, rotation_axes):
    rotations_c2to_c1 = eul_rot.as_np_array()
    c_frame_transform = FrameTransform(
        eul_rot, ref_translations, from_=eul_rot.from_, to_=eul_rot.to_
    )
    ref_translation_array = ref_translations.as_np_array()
    p_asset = c_frame_transform.transform_point(p_robot, from_="robot", to_="asset")
    transform = AlignFrames.align_frames(p_robot, p_asset, rotation_axes)

    assert np.allclose(
        transform.transform.translation.as_np_array(), ref_translation_array
    )

    assert np.allclose(transform.transform.euler.as_np_array(), rotations_c2to_c1)

    p_robot_noisy = PointList.from_array(
        p_robot.as_np_array()
        + np.clip(
            np.random.normal(np.zeros(p_robot.as_np_array().shape), 0.1), -0.1, 0.1
        ),
        frame="robot",
    )

    p_asset_noisy = PointList.from_array(
        p_asset.as_np_array()
        + np.clip(
            np.random.normal(np.zeros(p_asset.as_np_array().shape), 0.1), -0.1, 0.1
        ),
        frame="asset",
    )

    transform_noisy = AlignFrames.align_frames(
        p_robot_noisy, p_asset_noisy, rotation_axes
    )

    translation_arr_noise = transform_noisy.transform.translation.as_np_array()
    euler_arr_noise = transform_noisy.transform.euler.as_np_array()
    rotations = np.absolute(euler_arr_noise - rotations_c2to_c1)
    translations = np.absolute(translation_arr_noise - ref_translation_array)
    assert np.any(rotations > 0.3) == False
    assert np.any(translations > 0.4) == False


@pytest.mark.parametrize(
    "p_asset, p_robot,rotation_frame",
    [
        (
            PointList.from_array(np.array([[10, 0, 0], [5, 2, 0], [7, 5, 0]]), "asset"),
            PointList.from_array(np.array([[12, 0, 0], [5, 2, 0], [7, 5, 0]]), "robot"),
            "z",
        ),
        (
            PointList.from_array(np.array([[10, 0, 0], [5, 2, 0]]), "asset"),
            PointList.from_array(np.array([[13, 2, 0], [7, 4, 0]]), "robot"),
            "z",
        ),
        (
            PointList.from_array(np.array([[10, 0, 0], [5, 2, 0]]), "robot"),
            PointList.from_array(np.array([[11, 0, 0], [6, 2, 0]]), "asset"),
            "z",
        ),
        (
            PointList.from_array(
                np.array([[10, 0, 0], [10, 0, 0], [5, 2, 0], [7, 5, 0]]), "asset"
            ),
            PointList.from_array(
                np.array([[11, 0, 0], [11, 0, 0], [5, 2, 0], [7, 5, 0]]), "robot"
            ),
            "z",
        ),
        (
            PointList.from_array(np.array([[10, 0, 0], [5, 2, 0]]), "asset"),
            PointList.from_array(np.array([[11, 0, 0], [6, 2, 0]]), "robot"),
            "xyz",
        ),
    ],
)
def test_align_frames_exceptions(p_robot, p_asset, rotation_frame):
    with pytest.raises(ValueError):
        AlignFrames.align_frames(p_robot, p_asset, rotation_frame)
