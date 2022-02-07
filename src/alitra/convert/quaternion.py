from scipy.spatial.transform.rotation import Rotation

from alitra import Euler, Quaternion


def quaternion_to_euler(
    quaternion: Quaternion, sequence: str = "ZYX", degrees: bool = False
) -> Euler:
    """
    Transform a quaternion into Euler angles.
    :param quaternion: A Quaternion object.
    :param sequence: Rotation sequence for the Euler angles.
    :param degrees: Set to true if the resulting Euler angles should be in degrees. Default is radians.
    :return: Euler object.
    """
    rotation_object: Rotation = Rotation.from_quat(quaternion.as_np_array())
    euler: Euler = Euler.from_array(
        rotation_object.as_euler(sequence, degrees=degrees), frame="robot"
    )
    return euler
