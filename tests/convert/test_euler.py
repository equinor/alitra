import numpy as np
import pytest

from alitra import Euler, Quaternion
from alitra.convert.euler import euler_to_quaternion


@pytest.mark.parametrize(
    "euler, expected",
    [
        (
            Euler(psi=0, phi=0, theta=0, from_="robot", to_="asset"),
            Quaternion(x=0, y=0, z=0, w=1, frame="robot"),
        ),
        (
            Euler(
                psi=1.5707963,  # Z
                theta=0.5235988,  # Y
                phi=1.0471976,  # X
                from_="robot",
                to_="asset",
            ),
            Quaternion(x=0.1830127, y=0.5, z=0.5, w=0.6830127, frame="robot"),
        ),
    ],
)
def test_quaternion_to_euler(euler, expected):
    quaternion: Quaternion = euler_to_quaternion(
        euler=euler, sequence="ZYX", degrees=False
    )

    assert np.allclose(quaternion.as_np_array(), expected.as_np_array())
