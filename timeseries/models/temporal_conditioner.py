"""
TemporalNeuralConditioner

Prepares (B, T, N) binned time-series neural data for the U-Net cross-attention
layers by projecting N -> d_model and adding sinusoidal positional encoding
over the T (bin) dimension.

Shape flow:
  (B, T, N) -> input_proj  -> (B, T, d_model)
             -> + sine/cos PE (broadcast over B)
             -> dropout
             -> (B, T, d_model)   [ready for U-Net cross-attention as context]
"""

import math
import torch
import torch.nn as nn


class TemporalNeuralConditioner(nn.Module):
    def __init__(self, n_neurons: int, d_model: int, num_bins: int, dropout: float = 0.0):
        """
        Args:
            n_neurons:  Raw electrode/neuron count (N) - input feature size per bin.
            d_model:    Context embedding dimension. Must be even. Should equal
                        neural_embed_dim in the U-Net config.
            num_bins:   Number of temporal bins (T).
            dropout:    Dropout probability applied after PE addition.
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal positional encoding"

        self.d_model = d_model
        self.num_bins = num_bins

        # Project each time bin from electrode space -> embedding space.
        # nn.Linear applies to the last dimension, so it acts identically
        # across the T sequence dimension (shared weights per bin).
        self.input_proj = nn.Linear(n_neurons, d_model)

        # Pre-compute sinusoidal positional encoding over the T dimension.
        # Shape: (1, T, d_model) — the leading 1 broadcasts over the batch.
        pe = torch.zeros(1, num_bins, d_model)
        position = torch.arange(num_bins).unsqueeze(1).float()          # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                # (d_model/2,)
        pe[0, :, 0::2] = torch.sin(position * div_term)                 # even dims: sin
        pe[0, :, 1::2] = torch.cos(position * div_term)                 # odd dims:  cos
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N) — batch of T-bin neural time-series, N electrodes.
        Returns:
            out: (B, T, d_model) — context tokens for U-Net cross-attention.
        """
        # (B, T, N) -> (B, T, d_model)
        out = self.input_proj(x)
        # Add positional encoding; self.pe is (1, T, d_model) — broadcasts over B
        out = out + self.pe
        out = self.dropout(out)
        return out
