"""
Generate VAE-Reconstructed Timeseries HDF5
==========================================
Passes all train and test timeseries through a trained NeuralVAE (encode → decode
using mu, no reparameterization noise) and writes the reconstructed signals to a
new HDF5 file with the same keys and shape as the source.

The resulting HDF5 can then be used as the timeseries_h5_path for a DDPM config,
ensuring that the DDPM is trained and evaluated on the same distribution of signals
that eval_vae_diffusion.py feeds it at inference (VAE-smoothed, not raw).

Usage:
    python3 generate_vae_recon_ts.py \
        --vae_config  configs/ventral_vae_z128_beta001_e200.yaml \
        --output_h5   /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/ventral_stream_timeseries_vaerecon_beta001_e200.h5
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, 'models'))
from neural_vae import NeuralVAE, NeuralVAEDeep, NeuralVAEFlat

sys.path.insert(0, os.path.join(_here, '..', 'reconstruction'))
from utils import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

SPLITS = ['train_timeseries', 'test_timeseries']


def remap_legacy_vae_state_dict(state_dict):
    """Support checkpoints saved before res blocks were wrapped in nn.Sequential."""
    return {
        re.sub(r'\.(res\d+)\.(block|residual)\.', r'.\1.0.\2.', key): value
        for key, value in state_dict.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate VAE-reconstructed timeseries HDF5'
    )
    parser.add_argument('--vae_config', required=True, type=str,
                        help='Path to VAE YAML config')
    parser.add_argument('--output_h5', required=True, type=str,
                        help='Output HDF5 path for reconstructed timeseries')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for VAE inference (default: 256)')
    args = parser.parse_args()

    cfg = load_config(args.vae_config)
    dataset_cfg = cfg['dataset_params']
    model_cfg   = cfg['vae_params']
    train_cfg   = cfg['train_params']

    # ---- Build VAE ----
    vae_mc = model_cfg.get('model_class', 'default')
    if vae_mc == 'flat':
        vae_cls      = NeuralVAEFlat
        vae_extra_kw = {'T_win': dataset_cfg['T_win']}
    elif vae_mc == 'deep':
        vae_cls      = NeuralVAEDeep
        vae_extra_kw = {}
    else:
        vae_cls      = NeuralVAE
        vae_extra_kw = {}

    vae = vae_cls(
        num_neurons    = dataset_cfg['num_neurons'],
        enc_channels   = model_cfg['enc_channels'],
        z_channels     = model_cfg['z_channels'],
        kernel_size    = model_cfg.get('kernel_size', 3),
        num_groups     = model_cfg.get('num_groups', 8),
        num_res_blocks = model_cfg.get('num_res_blocks', 1),
        **vae_extra_kw,
    ).to(device)
    vae.eval()

    ckpt_path = os.path.join(train_cfg['ckpt_dir'], train_cfg['ckpt_name'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'VAE checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(remap_legacy_vae_state_dict(ckpt['vae']))
    print(f'Loaded VAE checkpoint: {ckpt_path}')

    src_h5_path = dataset_cfg['timeseries_h5_path']
    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)

    # T_win used during VAE training — reconstruct with matching window size so
    # the encoder/decoder operate in-distribution.  If T_win is absent from the
    # config (e.g. flat VAE), fall back to full-sequence reconstruction.
    T_win_train  = dataset_cfg.get('T_win', None)
    win_stride   = dataset_cfg.get('win_stride', 5)
    use_windowed = (T_win_train is not None) and (vae_mc != 'flat')
    if use_windowed:
        print(f'Windowed reconstruction: T_win={T_win_train}, stride={win_stride}')
    else:
        print('Full-sequence reconstruction (flat VAE or no T_win in config)')

    with h5py.File(src_h5_path, 'r') as src_f, \
         h5py.File(args.output_h5, 'w') as dst_f:

        for split in SPLITS:
            if split not in src_f:
                print(f'  Split "{split}" not found in source HDF5 — skipping.')
                continue

            data = src_f[split][()]           # (n, T, N)  float32
            n, T, N = data.shape
            print(f'\nProcessing {split}: shape {data.shape}')

            recon_all = np.zeros_like(data)   # (n, T, N)

            with torch.no_grad():
                for start in tqdm(range(0, n, args.batch_size),
                                  desc=f'  {split}', unit='batch'):
                    end  = min(start + args.batch_size, n)
                    x_np = data[start:end]     # (B, T, N)

                    # Replace NaNs with 0 (same as vae_dataset.py)
                    x_np = np.where(np.isnan(x_np), 0.0, x_np)

                    x = torch.tensor(x_np, dtype=torch.float32, device=device)
                    # VAE expects (B, N, T) — transpose
                    x = x.transpose(1, 2)      # (B, N, T)

                    if use_windowed:
                        # Slide T_win-sized windows over the full sequence and
                        # average overlapping reconstructions — this keeps the
                        # VAE operating at the same T it was trained on.
                        B = x.shape[0]
                        recon_sum   = torch.zeros(B, N, T, device=device)
                        count       = torch.zeros(T, device=device)
                        win_starts  = list(range(0, T - T_win_train + 1, win_stride))
                        # Ensure the last window always covers the sequence end
                        if not win_starts or win_starts[-1] + T_win_train < T:
                            win_starts.append(T - T_win_train)
                        for ws in win_starts:
                            we     = ws + T_win_train
                            x_win  = x[:, :, ws:we]           # (B, N, T_win)
                            mu_w, _logvar_w = vae.encode(x_win)
                            recon_w = vae.decode(mu_w, T_win=T_win_train)  # (B, N, T_win)
                            recon_sum[:, :, ws:we] += recon_w
                            count[ws:we] += 1
                        recon = (recon_sum / count.unsqueeze(0).unsqueeze(0))
                    else:
                        mu, _logvar = vae.encode(x)
                        # Use mu directly (deterministic) — no reparameterize noise
                        recon = vae.decode(mu, T_win=T)  # (B, N, T)

                    # Transpose back to (B, T, N)
                    recon = recon.transpose(1, 2).cpu().numpy()
                    recon_all[start:end] = recon

            dst_f.create_dataset(split, data=recon_all,
                                 compression='gzip', compression_opts=4)
            print(f'  Wrote {split}: shape {recon_all.shape}, '
                  f'mean={recon_all.mean():.4f}, std={recon_all.std():.4f}')

    print(f'\nDone. Output written to: {args.output_h5}')


if __name__ == '__main__':
    main()
