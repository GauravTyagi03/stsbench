"""
Neural Timeseries VAE
=====================
Compresses (B, N=315, T_win) neural signals into a compact latent space.

Architecture:
  Encoder: Conv1d blocks with one 2x downsample → (B, z_channels, T_win//2) mu, logvar
  Decoder: Mirror of encoder with ConvTranspose1d upsample → (B, N, T_win)
  No skip connections between encoder and decoder.
"""

import torch
import torch.nn as nn


class Conv1dResBlock(nn.Module):
    """
    GroupNorm → SiLU → Conv1d → GroupNorm → SiLU → Conv1d
    Residual: Conv1d(in, out, 1) if in_ch != out_ch
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.GroupNorm(min(num_groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.GroupNorm(min(num_groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
        )
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.residual(x)


class Encoder(nn.Module):
    """
    (B, N=315, T_win)
    → Conv1d(315, enc_ch[0])
    → ResBlock(enc_ch[0], enc_ch[0])
    → Conv1d(enc_ch[0], enc_ch[1], stride=2)   # downsample T
    → ResBlock(enc_ch[1], enc_ch[1])
    → GroupNorm + SiLU
    → mu_conv, logvar_conv: (B, z_channels, T_win//2)
    """
    def __init__(self, num_neurons=315, enc_channels=(128, 64),
                 z_channels=64, kernel_size=3, num_groups=8):
        super().__init__()
        c0, c1 = enc_channels
        padding = kernel_size // 2

        self.proj    = nn.Conv1d(num_neurons, c0, kernel_size, padding=padding)
        self.res0    = Conv1dResBlock(c0, c0, kernel_size, num_groups)
        self.down    = nn.Conv1d(c0, c1, kernel_size, padding=padding, stride=2)
        self.res1    = Conv1dResBlock(c1, c1, kernel_size, num_groups)
        self.norm    = nn.GroupNorm(min(num_groups, c1), c1)
        self.act     = nn.SiLU()
        self.mu_conv     = nn.Conv1d(c1, z_channels, 1)
        self.logvar_conv = nn.Conv1d(c1, z_channels, 1)

    def forward(self, x):
        # x: (B, N, T_win)
        h = self.proj(x)
        h = self.res0(h)
        h = self.down(h)
        h = self.res1(h)
        h = self.act(self.norm(h))
        return self.mu_conv(h), self.logvar_conv(h)


class Decoder(nn.Module):
    """
    z: (B, z_ch, T_win//2)
    → Conv1d(z_ch, enc_ch[1])
    → ResBlock(enc_ch[1], enc_ch[1])
    → ConvTranspose1d(enc_ch[1], enc_ch[0], k=2, stride=2)   # upsample T
    → ResBlock(enc_ch[0], enc_ch[0])
    → GroupNorm + SiLU → Conv1d(enc_ch[0], 315)
    Output: (B, N=315, T_win)
    """
    def __init__(self, num_neurons=315, enc_channels=(128, 64),
                 z_channels=64, kernel_size=3, num_groups=8):
        super().__init__()
        c0, c1 = enc_channels

        self.post_z  = nn.Conv1d(z_channels, c1, 1)
        self.res0    = Conv1dResBlock(c1, c1, kernel_size, num_groups)
        self.up      = nn.ConvTranspose1d(c1, c0, kernel_size=2, stride=2)
        self.res1    = Conv1dResBlock(c0, c0, kernel_size, num_groups)
        self.norm    = nn.GroupNorm(min(num_groups, c0), c0)
        self.act     = nn.SiLU()
        self.out_conv = nn.Conv1d(c0, num_neurons, kernel_size, padding=kernel_size // 2)

    def forward(self, z):
        # z: (B, z_ch, T_win//2)
        h = self.post_z(z)
        h = self.res0(h)
        h = self.up(h)
        h = self.res1(h)
        h = self.act(self.norm(h))
        return self.out_conv(h)   # (B, N, T_win)


class NeuralVAE(nn.Module):
    """
    Variational Autoencoder for neural timeseries.

    Input:  (B, N, T_win)   — N channels (neurons), T_win time steps
    Latent: (B, z_ch, T_win//2)
    Output: (B, N, T_win)

    forward() returns (recon, mu, logvar)
      recon: (B, T_win, N) — transposed for loss computation
    """
    def __init__(self, num_neurons=315, enc_channels=(128, 64),
                 z_channels=64, kernel_size=3, num_groups=8):
        super().__init__()
        self.encoder = Encoder(num_neurons, enc_channels, z_channels, kernel_size, num_groups)
        self.decoder = Decoder(num_neurons, enc_channels, z_channels, kernel_size, num_groups)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        """x: (B, N, T_win) → mu, logvar: (B, z_ch, T_win//2)"""
        return self.encoder(x)

    def decode(self, z, T_win=None):
        """
        z: (B, z_ch, T_latent) → (B, N, T_out)
        T_win: if provided, trims T_out to exactly T_win (handles odd input sizes).
        """
        out = self.decoder(z)
        if T_win is not None:
            out = out[:, :, :T_win]
        return out

    def forward(self, x):
        """
        x: (B, N, T_win)
        returns:
          recon:  (B, T_win, N)  — transposed, ready for MSE vs (B, T_win, N) target
          mu:     (B, z_ch, T_latent)
          logvar: (B, z_ch, T_latent)
        """
        T_win = x.shape[2]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_spatial = self.decode(z, T_win=T_win)  # (B, N, T_win) — trimmed
        recon = recon_spatial.transpose(1, 2)         # (B, T_win, N)
        return recon, mu, logvar


def vae_loss(recon, target, mu, logvar, beta=0.001):
    """
    recon:  (B, T_win, N)
    target: (B, T_win, N)
    Returns total_loss, recon_loss, kl_loss (all scalars).
    """
    recon_loss = torch.nn.functional.mse_loss(recon, target)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
