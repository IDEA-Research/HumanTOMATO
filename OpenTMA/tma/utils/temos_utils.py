from typing import Dict, List
import numpy as np
import torch
from torch import Tensor
import tma.utils.geometry as geometry


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None):
    """
    Converts lengths to a mask tensor.

    Args:
        lengths (List[int]): List of lengths.
        device (torch.device): The device on which the tensor will be allocated.
        max_len (int, optional): The maximum length. If None, the maximum length is set to the maximum value in lengths.

    Returns:
        Tensor: A tensor mask of shape (len(lengths), max_len).
    """

    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def detach_to_numpy(tensor):
    """
    Detaches a tensor and converts it to a numpy array.

    Args:
        tensor (Tensor): The tensor to detach and convert.

    Returns:
        ndarray: The detached tensor as a numpy array.
    """
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    """
    Removes padding from tensors according to the corresponding lengths.

    Args:
        tensors (List[Tensor]): List of tensors.
        lengths (List[int]): List of lengths.

    Returns:
        List[Tensor]: List of tensors with padding removed.
    """
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def nfeats_of(rottype):
    """
    Returns the number of features of a rotation type.

    Args:
        rottype (str): The rotation type.

    Returns:
        int: The number of features.

    Raises:
        TypeError: If the rotation type is not recognized.
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


def axis_angle_to(newtype, rotations):
    """
    Converts axis-angle rotations to another rotation type.

    Args:
        newtype (str): The new rotation type.
        rotations (Tensor): The rotations to convert.

    Returns:
        Tensor: The converted rotations.

    Raises:
        NotImplementedError: If the new rotation type is not recognized.
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


def matrix_to(newtype, rotations):
    """
    Converts matrix rotations to another rotation type.

    Args:
        newtype (str): The new rotation type.
        rotations (Tensor): The rotations to convert.

    Returns:
        Tensor: The converted rotations.

    Raises:
        NotImplementedError: If the new rotation type is not recognized.
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


def to_matrix(oldtype, rotations):
    """
    Converts rotations of a certain type to matrix rotations.

    Args:
        oldtype (str): The old rotation type.
        rotations (Tensor): The rotations to convert.

    Returns:
        Tensor: The converted rotations.

    Raises:
        NotImplementedError: If the old rotation type is not recognized.
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


# TODO: use a real subsampler..
def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


# TODO: use a real upsampler..
def upsample(motion, last_framerate, new_framerate):
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output


if __name__ == "__main__":
    motion = np.arange(105)
    submotion = motion[subsample(len(motion), 100.0, 12.5)]
    newmotion = upsample(submotion, 12.5, 100)

    print(newmotion)
