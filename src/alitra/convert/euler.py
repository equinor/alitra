from scipy.spatial.transform.rotation import Rotation

from alitra import Euler, Quaternion


def euler_to_quaternion(
    euler: Euler, sequence: str = "ZYX", degrees: bool = False
) -> Quaternion:
    """
    Transform a quaternion into Euler angles.
    :param euler: An Euler object.
    :param sequence: Rotation sequence for the Euler angles.
    :param degrees: Set to true if the provided Euler angles are in degrees. Default is radians.
    :return: Quaternion object.
    """
    rotation_object: Rotation = Rotation.from_euler(
        seq=sequence, angles=euler.as_np_array(), degrees=degrees
    )
    quaternion: Quaternion = Quaternion.from_array(
        rotation_object.as_quat(), frame="robot"
    )
    return quaternion
