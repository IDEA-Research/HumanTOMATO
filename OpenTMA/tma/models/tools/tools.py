import torch.nn as nn

def remove_padding(tensors, lengths):
    """
    Inputs:
        tensors (list): A list of tensors from which padding is to be removed.
        lengths (list): A list of integers representing the actual lengths of the tensors.

    This function removes padding from the tensors based on the actual lengths. 
    It returns a list of tensors with padding removed.

    Returns:
        list: A list of tensors with padding removed.
    """
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

class AutoParams(nn.Module):
    """
    This class is a subclass of nn.Module. 
    It is used to automatically set the parameters of a model. 
    It has two types of parameters: needed parameters and optional parameters. 
    Needed parameters must be provided when an instance of the class is created, 
    otherwise a ValueError is raised. 
    
    Optional parameters can be provided when an instance of the class is created, 
    otherwise they are set to their default values.
    """
    def __init__(self, **kargs):
        try:
            for param in self.needed_params:
                if param in kargs:
                    setattr(self, param, kargs[param])
                else:
                    raise ValueError(f"{param} is needed.")
        except :
            pass
            
        try:
            for param, default in self.optional_params.items():
                if param in kargs and kargs[param] is not None:
                    setattr(self, param, kargs[param])
                else:
                    setattr(self, param, default)
        except :
            pass
        super().__init__()


# taken from joeynmt repo
def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False
