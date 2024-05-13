from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from tma.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from tma.models.operator import PositionalEncoding
from tma.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from tma.models.operator.position_encoding import build_position_encoding
from tma.utils.temos_utils import lengths_to_mask

"""
vae
skip connection encoder 
skip connection decoder
mem for each decoder layer
"""


class VPosert(nn.Module):

    def __init__(self, cfg, **kwargs) -> None:

        super(VPosert, self).__init__()

        num_neurons = 512
        self.latentD = 256

        # self.num_joints = 21
        n_features = 196 * 263

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, n_features),
            ContinousRotReprDecoder(),
        )

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        q_z = self.encode(features)
        feats_rst = self.decode(q_z)
        return feats_rst, q_z

    def encode(self, pose_body, lengths: Optional[List[int]] = None):
        """
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        """
        q_z = self.encoder_net(pose_body)
        q_z_sample = q_z.rsample()
        return q_z_sample.unsqueeze(0), q_z

    def decode(self, Zin, lengths: Optional[List[int]] = None):
        bs = Zin.shape[0]
        Zin = Zin[0]

        prec = self.decoder_net(Zin)

        return prec


class BatchFlatten(nn.Module):

    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = "batch_flatten"

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ContinousRotReprDecoder(nn.Module):

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 196, 263)
        return reshaped_input


class NormalDistDecoder(nn.Module):

    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(
            self.mu(Xout), F.softplus(self.logvar(Xout))
        )
