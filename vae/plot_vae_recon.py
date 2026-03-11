"""
Visually verify VAE reconstruction quality on neural timeseries.

For N_SAMPLES test samples, plots:
  - Top row:    input heatmap (T x N)
  - Middle row: reconstructed heatmap (T x N)
  - Bottom row: difference (input - recon)

Usage:
    python plot_vae_recon.py --config configs/ventral_vae_z128_beta001.yaml [--n_samples 5]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'reconstruction'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'models'))

from utils import load_config, set_seed
from vae_dataset import SlidingWindowNeuralDataset
from neural_vae import NeuralVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    default='configs/ventral_vae_z128_beta001.yaml', type=str)
    parser.add_argument('--n_samples', default=5, type=int)
    parser.add_argument('--use_best',  action='store_true',
                        help='Load vae_ckpt_best.pth instead of vae_ckpt.pth')
    args = parser.parse_args()

    config       = load_config(args.config)
    dataset_cfg  = config['dataset_params']
    vae_cfg      = config['vae_params']
    train_cfg    = config['train_params']

    set_seed(train_cfg.get('seed', 42))

    # ---- load test dataset with T_win = num_bins (full window) ----
    T_win = dataset_cfg['num_bins']   # 15 — evaluate on full window
    test_dataset = SlidingWindowNeuralDataset(
        h5_path     = dataset_cfg['timeseries_h5_path'],
        split       = 'test',
        T_win       = T_win,
        win_stride  = T_win,
        use_sliding = False,
        num_neurons = dataset_cfg['num_neurons'],
    )

    # ---- load VAE ----
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

    n = min(args.n_samples, len(test_dataset))
    indices = np.linspace(0, len(test_dataset) - 1, n, dtype=int)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    model_name = config.get('model_name', 'vae')
    vmin, vmax = None, None  # set from first sample for consistent colorscale

    with torch.no_grad():
        for col, idx in enumerate(indices):
            x = test_dataset[idx].unsqueeze(0).to(device)   # (1, N, T_win)
            recon, mu, logvar = vae(x)

            inp   = x.squeeze().T.cpu().numpy()       # (T, N)
            rec   = recon.squeeze().cpu().numpy()     # (T, N)
            diff  = inp - rec

            if vmin is None:
                vmin, vmax = inp.min(), inp.max()

            for row, (data, title) in enumerate([(inp, 'Input'), (rec, 'Recon'), (diff, 'Diff')]):
                ax = axes[row, col]
                if row < 2:
                    im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax, cmap='RdBu_r')
                else:
                    abs_max = np.abs(diff).max()
                    im = ax.imshow(data, aspect='auto', vmin=-abs_max, vmax=abs_max, cmap='RdBu_r')
                ax.set_xlabel('Neuron')
                if col == 0:
                    ax.set_ylabel(f'{title}\nTime bin')
                else:
                    ax.set_title(title if row == 0 else '')
                if col == 0 and row == 0:
                    ax.set_title(f'Sample {idx}')
                elif row == 0:
                    ax.set_title(f'Sample {idx}')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # MSE summary across full test set
    all_mse = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x = test_dataset[i].unsqueeze(0).to(device)
            recon, _, _ = vae(x)
            mse = torch.nn.functional.mse_loss(recon, x.transpose(1, 2)).item()
            all_mse.append(mse)
    mean_mse = np.mean(all_mse)

    fig.suptitle(f'{model_name}  |  test MSE (all samples): {mean_mse:.5f}', fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(train_cfg['output_dir'], f'vae_recon_check.png')
    os.makedirs(train_cfg['output_dir'], exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    print(f'Mean test MSE across {len(test_dataset)} samples: {mean_mse:.5f}')


if __name__ == '__main__':
    main()
