"""
Visually verify VAE reconstruction quality on neural timeseries.

Produces two figures per run:

  Figure 1 — neuron_traces.png
    Grid of (n_samples rows) × (n_neurons cols).
    Each cell: input (solid blue) vs recon (dashed red) over T time bins.
    Lets you directly compare individual data points and their reconstruction.

  Figure 2 — scatter.png
    One scatter panel per sample: every (T × N) input value on the x-axis,
    its reconstruction on the y-axis.  Identity line shown; R² in title.
    Gives a global sense of reconstruction fidelity across all neurons/bins.

Usage:
    python plot_vae_recon.py --config configs/ventral_vae_z128_beta001.yaml
    python plot_vae_recon.py --config configs/ventral_vae_z128_beta001.yaml \
        --sample_idx 0 3 7 --neuron_idx 0 50 100 200 314
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'reconstruction'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'models'))

from utils import load_config, set_seed
from vae_dataset import SlidingWindowNeuralDataset
from neural_vae import NeuralVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_recon(vae, dataset, idx):
    """Returns (inp, recon) each of shape (T, N) as numpy arrays."""
    x = dataset[idx].unsqueeze(0).to(device)   # (1, N, T)
    with torch.no_grad():
        recon, _, _ = vae(x)
    inp   = x.squeeze().T.cpu().numpy()         # (T, N)
    recon = recon.squeeze().cpu().numpy()       # (T, N)
    return inp, recon


def plot_traces(vae, dataset, sample_indices, neuron_indices, out_path):
    """
    Grid: rows = samples, cols = neurons.
    Each cell: input (blue solid + dots) vs recon (red dashed + dots) over time.
    """
    n_s = len(sample_indices)
    n_n = len(neuron_indices)
    T   = dataset[0].shape[1]   # T_win
    t   = np.arange(T)

    fig, axes = plt.subplots(n_s, n_n, figsize=(3.5 * n_n, 2.8 * n_s), squeeze=False)

    for row, s_idx in enumerate(sample_indices):
        inp, recon = get_recon(vae, dataset, s_idx)
        for col, n_idx in enumerate(neuron_indices):
            ax = axes[row, col]
            ax.plot(t, inp[:, n_idx],   'b-o', ms=4, lw=1.5, label='input')
            ax.plot(t, recon[:, n_idx], 'r--o', ms=4, lw=1.5, label='recon')
            ax.set_title(f'sample {s_idx}, neuron {n_idx}', fontsize=8)
            ax.set_xlabel('time bin', fontsize=7)
            ax.set_xticks(t)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel('activity', fontsize=7)
            if row == 0 and col == n_n - 1:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Input (blue) vs Reconstruction (red) — per neuron time traces', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_scatter(vae, dataset, sample_indices, out_path):
    """
    One scatter panel per sample.
    X = input values (all T×N), Y = recon values.  Identity line + R².
    """
    n_s  = len(sample_indices)
    ncols = min(n_s, 5)
    nrows = (n_s + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for i, s_idx in enumerate(sample_indices):
        ax  = axes[i // ncols][i % ncols]
        inp, recon = get_recon(vae, dataset, s_idx)

        x_flat = inp.flatten()
        y_flat = recon.flatten()
        r, _   = pearsonr(x_flat, y_flat)
        r2     = r ** 2

        ax.scatter(x_flat, y_flat, s=2, alpha=0.3, color='steelblue', rasterized=True)
        lims = [min(x_flat.min(), y_flat.min()),
                max(x_flat.max(), y_flat.max())]
        ax.plot(lims, lims, 'k--', lw=1, label='identity')
        ax.set_title(f'sample {s_idx}  R²={r2:.3f}', fontsize=9)
        ax.set_xlabel('input', fontsize=8)
        ax.set_ylabel('recon', fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for j in range(len(sample_indices), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle('Input vs Reconstruction scatter (all T×N points per sample)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/ventral_vae_z128_beta001.yaml', type=str)
    parser.add_argument('--use_best',   action='store_true',
                        help='Load vae_ckpt_best.pth instead of vae_ckpt.pth')
    parser.add_argument('--sample_idx', nargs='+', type=int, default=None,
                        help='Which test samples to plot (default: 5 evenly spaced)')
    parser.add_argument('--neuron_idx', nargs='+', type=int, default=None,
                        help='Which neurons to show in trace plot (default: 5 evenly spaced)')
    parser.add_argument('--n_samples',  type=int, default=5)
    parser.add_argument('--n_neurons',  type=int, default=5)
    args = parser.parse_args()

    config      = load_config(args.config)
    dataset_cfg = config['dataset_params']
    vae_cfg     = config['vae_params']
    train_cfg   = config['train_params']

    set_seed(train_cfg.get('seed', 42))

    T_win = dataset_cfg['num_bins']
    test_dataset = SlidingWindowNeuralDataset(
        h5_path     = dataset_cfg['timeseries_h5_path'],
        split       = 'test',
        T_win       = T_win,
        win_stride  = T_win,
        use_sliding = False,
        num_neurons = dataset_cfg['num_neurons'],
    )

    vae = NeuralVAE(
        num_neurons  = dataset_cfg['num_neurons'],
        enc_channels = vae_cfg['enc_channels'],
        z_channels   = vae_cfg['z_channels'],
        kernel_size  = vae_cfg.get('kernel_size', 3),
        num_groups   = vae_cfg.get('num_groups', 8),
    ).to(device)

    ckpt_name = 'vae_ckpt_best.pth' if args.use_best else train_cfg['ckpt_name']
    ckpt_path = os.path.join(train_cfg['ckpt_dir'], ckpt_name)
    vae.load_state_dict(torch.load(ckpt_path, map_location=device)['vae'])
    vae.eval()

    N = dataset_cfg['num_neurons']
    n_test = len(test_dataset)

    sample_indices = (
        args.sample_idx
        if args.sample_idx is not None
        else list(np.linspace(0, n_test - 1, args.n_samples, dtype=int))
    )
    neuron_indices = (
        args.neuron_idx
        if args.neuron_idx is not None
        else list(np.linspace(0, N - 1, args.n_neurons, dtype=int))
    )

    out_dir = train_cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    plot_traces(
        vae, test_dataset, sample_indices, neuron_indices,
        os.path.join(out_dir, 'vae_neuron_traces.png'),
    )
    plot_scatter(
        vae, test_dataset, sample_indices,
        os.path.join(out_dir, 'vae_scatter.png'),
    )


if __name__ == '__main__':
    main()
