import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from tma.models.operator import PositionalEncoding
from tma.utils.temos_utils import lengths_to_mask


class GRUDecoder(pl.LightningModule):
    """
    This class is a decoder module for features using a GRU-based architecture.

    Args:
        nfeats (int): The number of features in the input.
        latent_dim (int, optional): The dimensionality of the latent space. Defaults to 256.
        num_layers (int, optional): The number of layers in the GRU model. Defaults to 4.
    """

    def __init__(
        self, nfeats: int, latent_dim: int = 256, num_layers: int = 4, **kwargs
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        # Embedding layer to transform the input
        self.emb_layer = nn.Linear(latent_dim + 1, latent_dim)

        # GRU layer
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers)

        # Final linear layer
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths: List[int]):
        """
        Forward pass for the decoder.

        Args:
            z (Tensor): The input tensor.
            lengths (List[int]): The lengths of the sequences.

        Returns:
            Tensor: The output features.
        """

        # Create a mask based on the lengths
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        lengths = torch.tensor(lengths, device=z.device)

        # Repeat the input
        z = z[None].repeat((nframes, 1, 1))

        # Add time information to the input
        time = mask * 1 / (lengths[..., None] - 1)
        time = (time[:, None] * torch.arange(time.shape[1], device=z.device))[:, 0]
        time = time.T[..., None]
        z = torch.cat((z, time), 2)

        # emb to latent space again
        z = self.emb_layer(z)

        # pass to gru
        z = self.gru(z)[0]
        output = self.final_layer(z)

        # zero for padded area
        output[~mask.T] = 0

        # Pytorch GRU: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)

        return feats
