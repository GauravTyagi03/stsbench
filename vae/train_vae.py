"""
Train Neural Timeseries VAE
============================
Config-driven training loop for NeuralVAE on ventral-stream neural data.

Usage:
    python train_vae.py --config configs/ventral_vae_z64_beta001.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# reuse load_config and set_seed from reconstruction/utils.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reconstruction'))
from utils import load_config, set_seed

from vae_dataset import SlidingWindowNeuralDataset
from models.neural_vae import NeuralVAE, vae_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def main():
    parser = argparse.ArgumentParser(description='Train Neural Timeseries VAE')
    parser.add_argument('--config', default='configs/ventral_vae_z64_beta001.yaml', type=str)
    args = parser.parse_args()

    config       = load_config(args.config)
    dataset_cfg  = config['dataset_params']
    vae_cfg      = config['vae_params']
    train_cfg    = config['train_params']

    set_seed(train_cfg.get('seed', 42))

    # ---- datasets ----
    train_dataset = SlidingWindowNeuralDataset(
        h5_path      = dataset_cfg['timeseries_h5_path'],
        split        = 'train',
        T_win        = dataset_cfg['T_win'],
        win_stride   = dataset_cfg['win_stride'],
        use_sliding  = True,
    )
    val_dataset = SlidingWindowNeuralDataset(
        h5_path      = dataset_cfg['timeseries_h5_path'],
        split        = 'test',
        T_win        = dataset_cfg['T_win'],
        win_stride   = dataset_cfg['win_stride'],
        use_sliding  = False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg['batch_size'],
        shuffle     = True,
        num_workers = 8,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = train_cfg['batch_size'],
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )

    # ---- model ----
    vae = NeuralVAE(
        num_neurons  = dataset_cfg['num_neurons'],
        enc_channels = vae_cfg['enc_channels'],
        z_channels   = vae_cfg['z_channels'],
        kernel_size  = vae_cfg.get('kernel_size', 3),
        num_groups   = vae_cfg.get('num_groups', 8),
    ).to(device)

    optimizer = Adam(vae.parameters(), lr=train_cfg['lr'])
    beta      = train_cfg['beta']
    epochs    = train_cfg['epochs']

    # ---- logging ----
    os.makedirs(train_cfg['output_dir'], exist_ok=True)
    os.makedirs(train_cfg['ckpt_dir'],   exist_ok=True)
    log_path = os.path.join(train_cfg['output_dir'], train_cfg['log_name'])
    if os.path.exists(log_path):
        os.remove(log_path)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # ---- train ----
        vae.train()
        train_losses, train_recon, train_kl = [], [], []

        for x in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [train]'):
            x = x.to(device)                   # (B, N, T_win)
            target = x.transpose(1, 2)         # (B, T_win, N)

            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            total, recon_l, kl_l = vae_loss(recon, target, mu, logvar, beta)
            total.backward()
            optimizer.step()

            train_losses.append(total.item())
            train_recon.append(recon_l.item())
            train_kl.append(kl_l.item())

        # ---- val ----
        vae.eval()
        val_losses, val_recon, val_kl = [], [], []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                target = x.transpose(1, 2)
                recon, mu, logvar = vae(x)
                total, recon_l, kl_l = vae_loss(recon, target, mu, logvar, beta)
                val_losses.append(total.item())
                val_recon.append(recon_l.item())
                val_kl.append(kl_l.item())

        log_line = (
            f'Epoch {epoch:03d} | '
            f'train_total={np.mean(train_losses):.6f}  '
            f'train_recon={np.mean(train_recon):.6f}  '
            f'train_kl={np.mean(train_kl):.6f}  || '
            f'val_total={np.mean(val_losses):.6f}  '
            f'val_recon={np.mean(val_recon):.6f}  '
            f'val_kl={np.mean(val_kl):.6f}'
        )
        print(log_line)
        with open(log_path, 'a') as f:
            f.write(log_line + '\n')

        # ---- checkpoint ----
        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = os.path.join(train_cfg['ckpt_dir'], train_cfg['ckpt_name'])
            torch.save({'vae': vae.state_dict()}, ckpt_path)

        val_mean = np.mean(val_losses)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_path = os.path.join(train_cfg['ckpt_dir'], 'vae_ckpt_best.pth')
            torch.save({'vae': vae.state_dict()}, best_path)

    print('Training complete.')


if __name__ == '__main__':
    main()
