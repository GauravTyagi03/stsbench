"""
Plot VAE training and validation loss curves from log files.

For each log file, produces a figure with one subplot per tracked metric
(total, recon, kl, and temporal if present).  Both train and val curves are
overlaid on each subplot so convergence is immediately visible.

Output is saved as  <run_name>_training_curves.png  in the same directory as
the log file, where run_name is the parent directory name.

Usage:
    python plot_vae_training.py --logs path/to/vae_training_log.txt [...]
    python plot_vae_training.py --logs logs/vae_z128_beta001/vae_training_log.txt \
                                        logs/vae_z128_twin10_temporal_alpha05/vae_training_log.txt
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_KV_RE  = re.compile(r'(\w+)=([-\d.eE+]+)')
_EP_RE  = re.compile(r'Epoch\s+(\d+)')

METRICS = ['total', 'recon', 'temporal', 'kl']


def parse_log(log_path):
    """
    Returns a dict:
      {
        'epochs': [1, 2, ...],
        'train_total': [...], 'val_total': [...],
        'train_recon': [...], 'val_recon': [...],
        'train_kl':    [...], 'val_kl':    [...],
        'train_temporal': [...], 'val_temporal': [...],   # only if present
      }
    Handles both old (no temporal) and new (with temporal) log formats.
    """
    data = {'epochs': []}
    for m in METRICS:
        data[f'train_{m}'] = []
        data[f'val_{m}']   = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep_match = _EP_RE.search(line)
            if ep_match is None:
                continue
            data['epochs'].append(int(ep_match.group(1)))

            kv = dict(_KV_RE.findall(line))
            for m in METRICS:
                data[f'train_{m}'].append(float(kv.get(f'train_{m}', 'nan')))
                data[f'val_{m}'].append(float(kv.get(f'val_{m}',   'nan')))

    # convert to arrays
    for key in list(data.keys()):
        data[key] = np.array(data[key], dtype=float)

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(log_path):
    run_name = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
    data     = parse_log(log_path)
    epochs   = data['epochs']

    if len(epochs) == 0:
        print(f'  [skip] no epochs parsed from {log_path}')
        return

    # decide which metrics to show (skip temporal if all NaN)
    active_metrics = []
    for m in METRICS:
        train_vals = data[f'train_{m}']
        if not np.all(np.isnan(train_vals)):
            active_metrics.append(m)

    n = len(active_metrics)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for i, m in enumerate(active_metrics):
        ax = axes[i // ncols][i % ncols]
        train_vals = data[f'train_{m}']
        val_vals   = data[f'val_{m}']

        ax.plot(epochs, train_vals, color='steelblue', lw=1.5, label='train')
        ax.plot(epochs, val_vals,   color='darkorange', lw=1.5, linestyle='--', label='val')

        # mark best val epoch
        best_ep_idx = int(np.nanargmin(val_vals))
        ax.axvline(epochs[best_ep_idx], color='darkorange', lw=0.8, linestyle=':',
                   alpha=0.7, label=f'best val ep {epochs[best_ep_idx]}')

        ax.set_title(f'{m} loss', fontsize=10)
        ax.set_xlabel('epoch', fontsize=9)
        ax.set_ylabel('loss', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    final_val_recon = data['val_recon'][-1] if len(data['val_recon']) else float('nan')
    fig.suptitle(f'{run_name}  (final val_recon={final_val_recon:.5f})', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(log_path), f'{run_name}_training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True,
                        help='Paths to vae_training_log.txt files')
    args = parser.parse_args()

    for log_path in args.logs:
        if not os.path.isfile(log_path):
            print(f'  [skip] not found: {log_path}')
            continue
        print(f'Plotting: {log_path}')
        plot_training(log_path)


if __name__ == '__main__':
    main()
