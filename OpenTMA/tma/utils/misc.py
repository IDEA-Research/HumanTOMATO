import torch


def to_numpy(tensor):
    """
    Converts a PyTorch tensor to a numpy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        ndarray (numpy.ndarray): The converted numpy array.

    Raises:
    """
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    """
    Converts a numpy array to a PyTorch tensor.

    Args:
        ndarray (numpy.ndarray): The numpy array to convert.

    Returns:
        tensor (torch.Tensor): The converted PyTorch tensor.

    Raises:
        ValueError: If the input is not a numpy array.
    """
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    """
    Exits the program cleanly by handling the SystemExit exception.

    No input arguments or return values.
    """
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
