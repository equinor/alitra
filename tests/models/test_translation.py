import numpy as np
import pytest

from alitra import Translation


def test_translation_array(robot_frame, asset_frame):
    expected_array = np.array([1, 1, 1])
    translation: Translation = Translation.from_array(
        expected_array, from_=robot_frame, to_=asset_frame
    )
    assert np.allclose(expected_array, translation.to_array())


def test_translation_invalid_array(robot_frame, asset_frame):
    with pytest.raises(ValueError):
        Translation.from_array(np.array([1, 1]), from_=robot_frame, to_=asset_frame)
