"""
TemporalNeuralConditioner

Prepares (B, T, N) binned time-series neural data for the U-Net cross-attention
layers by projecting N -> d_model and adding sinusoidal positional encoding
over the T (bin) dimension.

Shape flow:
  (B, T, N) -> input_proj  -> (B, T, d_model)
             -> + sine/cos PE (broadcast over B)
             -> TemporalEncoder (optional; residual)
             -> dropout
             -> (B, T, d_model)   [ready for U-Net cross-attention as context]

temporal_encoder_type:
  'none'   — no additional encoder (original behaviour)
  'conv1d' — Conv1d(kernel=3, padding=1) + GELU, residual
  'gru'    — GRU(num_layers=1), residual
"""

import math
import torch
import torch.nn as nn


class TemporalNeuralConditioner(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        num_bins: int,
        dropout: float = 0.0,
        temporal_encoder_type: str = 'none',
    ):
        """
        Args:
            n_neurons:             Raw electrode/neuron count (N) - input feature size per bin.
            d_model:               Context embedding dimension. Must be even. Should equal
                                   neural_embed_dim in the U-Net config.
            num_bins:              Number of temporal bins (T).
            dropout:               Dropout probability applied after PE addition.
            temporal_encoder_type: One of 'none', 'conv1d', 'gru'.
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal positional encoding"
        assert temporal_encoder_type in ('none', 'conv1d', 'gru'), (
            f"temporal_encoder_type must be 'none', 'conv1d', or 'gru'; got {temporal_encoder_type!r}"
        )

        self.d_model = d_model
        self.num_bins = num_bins
        self.temporal_encoder_type = temporal_encoder_type

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

        # Optional temporal encoder
        if temporal_encoder_type == 'conv1d':
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
        elif temporal_encoder_type == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )

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
        out_pe = out + self.pe

        if self.temporal_encoder_type == 'conv1d':
            # Conv1d expects (B, C, L); transpose T and d dims
            conv_in = out_pe.transpose(1, 2)          # (B, d_model, T)
            conv_out = self.temporal_encoder(conv_in)  # (B, d_model, T)
            out = out_pe + conv_out.transpose(1, 2)    # residual, back to (B, T, d_model)
        elif self.temporal_encoder_type == 'gru':
            gru_out, _ = self.temporal_encoder(out_pe)  # (B, T, d_model)
            out = out_pe + gru_out                       # residual
        else:
            out = out_pe

        out = self.dropout(out)
        return out
