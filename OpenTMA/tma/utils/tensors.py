import torch


def lengths_to_mask(lengths):
    """
    Converts lengths to a mask tensor.

    Args:
        lengths (Tensor): A tensor of lengths.

    Returns:
        Tensor: A tensor mask of shape (len(lengths), max_len).
    """
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    """
    Collates a batch of tensors by padding them to the same size.

    Args:
        batch (List[Tensor]): A list of tensors.

    Returns:
        Tensor: A tensor of shape (len(batch), max_size).
    """
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    """
    Collates a batch of data and labels, and generates a mask tensor.

    Args:
        batch (List[Tuple[Tensor, Tensor]]): A list of tuples, each containing a tensor of data and a tensor of labels.

    Returns:
        dict: A dictionary containing the collated data, labels, mask, and lengths.
    """
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]

    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)

    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, 'lengths': lenbatchTensor}
    return batch


# slow version with padding
def collate_data3d_slow(batch):
    """
    Collates a batch of 3D data by padding them to the same size.

    Args:
        batch (List[dict]): A list of dictionaries, each containing a tensor of 3D data.

    Returns:
        dict: A dictionary containing the collated 3D data.
    """
    batchTensor = {}
    for key in batch[0].keys():
        databatch = [b[key] for b in batch]
        batchTensor[key] = collate_tensors(databatch)

    batch = batchTensor
    return batch


def collate_data3d(batch):
    """
    Collates a batch of 3D data by stacking them along a new dimension.

    Args:
        batch (List[dict]): A list of dictionaries, each containing a tensor of 3D data.

    Returns:
        dict: A dictionary containing the collated 3D data.
    """
    batchTensor = {}
    for key in batch[0].keys():
        databatch = [b[key] for b in batch]
        if key == "paths":
            batchTensor[key] = databatch
        else:
            batchTensor[key] = torch.stack(databatch, axis=0)

    batch = batchTensor
    return batch
