import numpy as np
import pytest

from alitra import Euler, Quaternion
from alitra.convert import quaternion_to_euler


@pytest.mark.parametrize(
    "quaternion, expected",
    [
        (
            Quaternion(x=0, y=0, z=0, w=1, frame="robot"),
            Euler(psi=0, phi=0, theta=0, from_="robot", to_="asset"),
        ),
        (
            Quaternion(x=0.1830127, y=0.5, z=0.5, w=0.6830127, frame="robot"),
            Euler(
                psi=1.5707963,  # Z
                theta=0.5235988,  # Y
                phi=1.0471976,  # X
                from_="robot",
                to_="asset",
            ),
        ),
    ],
)
def test_quaternion_to_euler(quaternion, expected):
    euler_angles: Euler = quaternion_to_euler(
        quaternion=quaternion, sequence="ZYX", degrees=False
    )
    assert np.allclose(euler_angles.as_np_array(), expected.as_np_array())
