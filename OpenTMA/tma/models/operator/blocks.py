import torch
import torch.nn as nn
import torch.nn.functional as F
from tma.models.operator import AdaptiveInstanceNorm1d


class MLP(nn.Module):
    """
    This class is a subclass of nn.Module. 
    It implements a Multilayer Perceptron (MLP) with configurable dimensions and activation functions.
    """
    def __init__(self, cfg, out_dim, is_init):
        """
        Inputs:
            cfg (dict): The configuration dictionary.
            out_dim (int): The dimension of the output.
            is_init (bool): Whether to initialize the weights.

        This function is the constructor of the MLP class. It initializes the MLP with the given configuration and output dimension. If is_init is True, it also initializes the weights.
        """
        super(MLP, self).__init__()
        dims = cfg.MODEL.MOTION_DECODER.MLP_DIM
        n_blk = len(dims)
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim, norm='none', acti='none')
        self.model = nn.Sequential(*layers)

        if is_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Inputs:
            x (Tensor): The input tensor.

        This function applies the MLP to the input tensor and returns the output tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.model(x.view(x.size(0), -1))


def ZeroPad1d(sizes):
    """
    Inputs:
        sizes (tuple): The padding sizes.

    This function returns a 1D zero padding layer with the given sizes.

    Returns:
        nn.ConstantPad1d: The 1D zero padding layer.
    """
    return nn.ConstantPad1d(sizes, 0)


def get_acti_layer(acti='relu', inplace=True):
    """
    Inputs:
        acti (str): The activation function. Default is 'relu'.
        inplace (bool): Whether to apply the activation function in-place. Default is True.

    This function returns a list containing the activation layer.

    Returns:
        list: A list containing the activation layer.
    """

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):
    """
    Inputs:
        norm (str): The normalization function. Default is 'none'.
        norm_dim (int): The dimension of the normalization.

    This function returns a list containing the normalization layer.

    Returns:
        list: A list containing the normalization layer.
    """

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    """
    Inputs:
        dropout (float): The dropout rate.

    This function returns a list containing the dropout layer.

    Returns:
        list: A list containing the dropout layer.
    """
    
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvLayers(kernel_size,
               in_channels,
               out_channels,
               stride=1,
               pad_type='reflect',
               use_bias=True):
    """
    Inputs:
        kernel_size (int): The size of the kernel.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int): The stride of the convolution. Default is 1.
        pad_type (str): The type of padding. Default is 'reflect'.
        use_bias (bool): Whether to use bias. Default is True.

    This function returns a list containing the padding and convolution layers.

    Returns:
        list: A list containing the padding and convolution layers.
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [
        pad((pad_l, pad_r)),
        nn.Conv1d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  bias=use_bias)
    ]


def ConvBlock(kernel_size,
              in_channels,
              out_channels,
              stride=1,
              pad_type='reflect',
              dropout=None,
              norm='none',
              acti='lrelu',
              acti_first=False,
              use_bias=True,
              inplace=True):
    """
    Inputs:
        kernel_size (int): The size of the kernel.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int): The stride of the convolution. Default is 1.
        pad_type (str): The type of padding. Default is 'reflect'.
        dropout (float): The dropout rate.
        norm (str): The normalization function. Default is 'none'.
        acti (str): The activation function. Default is 'lrelu'.
        acti_first (bool): Whether to apply the activation function first. Default is False.
        use_bias (bool): Whether to use bias. Default is True.
        inplace (bool): Whether to apply the activation function in-place. Default is True.

    This function returns a list containing the convolution block layers.

    Returns:
        list: A list containing the convolution block layers.
    """

    layers = ConvLayers(kernel_size,
                        in_channels,
                        out_channels,
                        stride=stride,
                        pad_type=pad_type,
                        use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):
    """
    Inputs:
        in_dim (int): The dimension of the input.
        out_dim (int): The dimension of the output.
        dropout (float): The dropout rate.
        norm (str): The normalization function. Default is 'none'.
        acti (str): The activation function. Default is 'relu'.

    This function returns a list containing the linear block layers.

    Returns:
        list: A list containing the linear block layers.
    """
     
    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers
