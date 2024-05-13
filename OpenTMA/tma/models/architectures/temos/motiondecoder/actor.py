import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from tma.models.operator import PositionalEncoding
from tma.utils.temos_utils import lengths_to_mask


class ActorAgnosticDecoder(pl.LightningModule):
    """
    This class is a decoder module for actor-agnostic features. It uses a transformer-based architecture for decoding.

    Args:
        nfeats (int): The number of features in the input.
        latent_dim (int, optional): The dimensionality of the latent space. Defaults to 256.
        ff_size (int, optional): The dimensionality of the feedforward network model. Defaults to 1024.
        num_layers (int, optional): The number of sub-encoder-layers in the transformer model. Defaults to 4.
        num_heads (int, optional): The number of heads in the multiheadattention models. Defaults to 4.
        dropout (float, optional): The dropout value. Defaults to 0.1.
        activation (str, optional): The activation function of intermediate layer, relu or gelu. Defaults to "gelu".
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        # Transformer decoder
        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

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

        z = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
