import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from tma.models.operator import PositionalEncoding
from tma.utils.temos_utils import lengths_to_mask


class GRUEncoder(pl.LightningModule):
    """
    This class is a GRU encoder for encoding input features.

    Attributes:
    - skel_embedding: a linear layer for embedding the input features.
    - gru: a GRU layer for encoding the embedded features.
    - mu: a linear layer for generating the mean of the latent distribution (only if VAE is used).
    - logvar: a linear layer for generating the log variance of the latent distribution (only if VAE is used).
    - final: a linear layer for generating the final output (only if VAE is not used).

    Methods:
    - __init__: initializes the GRUEncoder object with the given parameters.
    - forward: encodes the input features and returns the encoded output.
    """

    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        num_layers: int = 4,
        **kwargs
    ):
        """
        Initializes the GRUEncoder object with the given parameters.

        Inputs:
        - nfeats: the number of input features.
        - vae: a flag indicating whether to use a Variational Autoencoder (VAE).
        - latent_dim: the dimension of the latent space.
        - num_layers: the number of layers in the GRU.

        Outputs: None
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        input_feats = nfeats

        # Embed the input features
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        # Initialize the GRU layer
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers)

        # Action agnostic: only one set of params
        if vae:
            self.mu = nn.Linear(latent_dim, latent_dim)
            self.logvar = nn.Linear(latent_dim, latent_dim)
        else:
            self.final = nn.Linear(latent_dim, latent_dim)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        """
        Encodes the input features and returns the encoded output.

        Inputs:
        - features: a tensor of input features.
        - lengths: a list of lengths of the input features.

        Outputs: the encoded output.
        """
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Get all the output of the gru
        x = self.gru(x)[0]

        # Put back the batch dimention first
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Extract the last valid input
        x = x[
            tuple(
                torch.stack(
                    (
                        torch.arange(bs, device=x.device),
                        torch.tensor(lengths, device=x.device) - 1,
                    )
                )
            )
        ]

        if self.hparams.vae:
            mu = self.mu(x)
            logvar = self.logvar(x)
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            return torch.distributions.Normal(mu, std)
        else:
            return self.final(x)
