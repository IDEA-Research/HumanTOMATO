import tma.utils.geometry as geometry


# This function returns the number of features for a given rotation type.
def nfeats_of(rottype):
    """
    Parameters:
    rottype (str): The type of rotation.

    Returns:
    int: The number of features for the rotation type.
    """
    if rottype in ["rotvec", "axisangle"]:
        return 3
    elif rottype in ["rotquat", "quaternion"]:
        return 4
    elif rottype in ["rot6d", "6drot", "rotation6d"]:
        return 6
    elif rottype in ["rotmat"]:
        return 9
    else:
        return TypeError("This rotation type doesn't have features.")

# This function converts axis-angle rotations to another rotation type.


def axis_angle_to(newtype, rotations):
    """
    Parameters:
    newtype (str): The new type of rotation.
    rotations (np.array): The axis-angle rotations.

    Returns:
    np.array: The rotations converted to the new type.
    """
    if newtype in ["matrix"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.axis_angle_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        return rotations
    else:
        raise NotImplementedError

# This function converts matrix rotations to another rotation type.


def matrix_to(newtype, rotations):
    """
    Parameters:
    newtype (str): The new type of rotation.
    rotations (np.array): The matrix rotations.

    Returns:
    np.array: The rotations converted to the new type.
    """
    if newtype in ["matrix"]:
        return rotations
    if newtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 9))
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = geometry.matrix_to_axis_angle(rotations)
        return rotations
    else:
        raise NotImplementedError

# This function converts rotations of a given type to a matrix.


def to_matrix(oldtype, rotations):
    """
    Parameters:
    oldtype (str): The old type of rotation.
    rotations (np.array): The rotations.

    Returns:
    np.array: The rotations converted to a matrix.
    """
    if oldtype in ["matrix"]:
        return rotations
    if oldtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 3, 3))
        return rotations
    elif oldtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = geometry.quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError
