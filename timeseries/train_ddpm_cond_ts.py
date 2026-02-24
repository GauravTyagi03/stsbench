"""
Train Conditional DDPM with Timeseries Neural Conditioning
===========================================================
Mirrors reconstruction/train_ddpm_cond.py with the following changes:

1. Loads binned timeseries data  (B, T, N)  via dataloader_ts.py
2. Instantiates TemporalNeuralConditioner which projects (B, T, N) -> (B, T, d_model)
   and adds sinusoidal positional encoding over the T dimension
3. Classifier-free guidance dropout zeroes the full (B, T, N) input before projection
4. Optimizer covers both the U-Net and the TemporalNeuralConditioner jointly
5. Checkpoint saves both model and temporal_cond state dicts under separate keys
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

# ---- shared modules live in reconstruction/ ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reconstruction'))
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import load_config, set_seed

# ---- timeseries-specific modules ----
from dataloader_ts import get_timeseries_stimulus_datasets, get_dataloaders
from models.temporal_conditioner import TemporalNeuralConditioner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    parser = argparse.ArgumentParser(description='Arguments for timeseries DDPM training')
    parser.add_argument('--config', default='configs/dorsal_stream_diffusion_ts.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    name                    = config['name']
    modality                = config.get('modality', 'image')
    diffusion_config        = config['diffusion_params']
    dataset_config          = config['dataset_params']
    diffusion_model_config  = config['ldm_params']
    autoencoder_model_config= config['autoencoder_params']
    train_config            = config['train_params']
    condition_config        = diffusion_model_config['condition_config']
    neural_cond_config      = condition_config['neural_condition_config']

    set_seed()

    # ---- noise scheduler (unchanged) ----
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    condition_types  = condition_config['condition_types']
    assert 'neural' in condition_types, "Timeseries training requires neural conditioning"

    num_bins         = neural_cond_config['num_bins']
    num_neurons      = neural_cond_config['num_neurons']     # raw electrode count
    temporal_d_model = neural_cond_config['temporal_d_model']
    temporal_dropout = neural_cond_config.get('temporal_dropout', 0.0)
    cond_drop_prob   = neural_cond_config['cond_drop_prob']

    # neural_embed_dim seen by the U-Net must equal temporal_d_model (conditioner output)
    assert neural_cond_config['neural_embed_dim'] == temporal_d_model, (
        f"neural_embed_dim ({neural_cond_config['neural_embed_dim']}) must equal "
        f"temporal_d_model ({temporal_d_model}) — the U-Net cross-attention "
        f"context_dim is set from neural_embed_dim."
    )

    # ---- TemporalNeuralConditioner ----
    temporal_cond = TemporalNeuralConditioner(
        n_neurons=num_neurons,
        d_model=temporal_d_model,
        num_bins=num_bins,
        dropout=temporal_dropout,
    ).to(device)

    # ---- dataset ----
    train_dataset, test_dataset = get_timeseries_stimulus_datasets(
        name=name,
        stimulus_size=dataset_config['im_size'],
        num_neurons=num_neurons,
        timeseries_h5_path=dataset_config['timeseries_h5_path'],
        modality=modality,
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, test_dataset,
        batch_size=train_config['ldm_batch_size'],
        val_prop=0, seed=42, num_workers=8,
    )

    # ---- U-Net (unchanged architecture) ----
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config,
    ).to(device)
    model.train()

    # ---- frozen VQ-VAE ----
    vae = VQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config,
    ).to(device)
    vae.eval()
    # vqvae_ckpt_dir allows the VQ-VAE checkpoint to live in reconstruction/checkpoints
    vqvae_ckpt_dir = train_config.get('vqvae_ckpt_dir', train_config['ckpt_dir'])
    vae.load_state_dict(torch.load(
        os.path.join(vqvae_ckpt_dir, train_config['vqvae_autoencoder_ckpt_name']),
        map_location=device,
    ))
    for param in vae.parameters():
        param.requires_grad = False

    # ---- logging ----
    os.makedirs(train_config['output_dir'], exist_ok=True)
    os.makedirs(train_config['ckpt_dir'],   exist_ok=True)
    log_file = os.path.join(train_config['output_dir'], train_config['ldm_log_name'])
    if os.path.exists(log_file):
        os.remove(log_file)

    # ---- optimizer: U-Net + TemporalNeuralConditioner jointly ----
    num_epochs = train_config['ldm_epochs']
    optimizer  = Adam(
        list(model.parameters()) + list(temporal_cond.parameters()),
        lr=train_config['ldm_lr'],
    )
    criterion = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        model.train()
        temporal_cond.train()

        for im, cond_input in tqdm(train_loader):
            optimizer.zero_grad()

            im = im.float().to(device)
            with torch.no_grad():
                im, _ = vae.encode(im)

            ########### Timeseries Neural Conditioning ###########
            # cond_input: (B, T, N) — already time-binned by the dataloader
            neural_condition = cond_input.to(device)

            # Classifier-free guidance dropout: zero out entire samples in-place.
            # The conditioner sees zero input -> outputs PE-only tokens (no neural signal),
            # matching the null conditioning convention used during sampling.
            if cond_drop_prob > 0:
                neural_drop_mask = (
                    torch.zeros(im.shape[0], device=device).float().uniform_(0, 1)
                    < cond_drop_prob
                )
                neural_condition[neural_drop_mask] = 0.0  # (B, T, N) zero rows

            # Project N -> d_model and add positional encoding (needs gradients)
            neural_condition = temporal_cond(neural_condition)   # (B, T, d_model)
            cond_input = neural_condition
            ######################################################

            noise     = torch.randn_like(im).to(device)
            t         = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            noisy_im  = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1, np.mean(losses)))

        with open(log_file, 'a') as f:
            f.write('Finished epoch:{} | Loss : {:.4f}\n'.format(
                epoch_idx + 1, np.mean(losses)
            ))

        if epoch_idx % 5 == 0 or epoch_idx == num_epochs - 1:
            torch.save(
                {
                    'model':        model.state_dict(),
                    'temporal_cond': temporal_cond.state_dict(),
                },
                os.path.join(train_config['ckpt_dir'], train_config['ldm_ckpt_name']),
            )

    print('Done Training ...')


if __name__ == '__main__':
    main()
