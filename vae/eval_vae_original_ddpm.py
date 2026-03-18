"""
Evaluate VAE-decoded conditioning with the original static-conditioning Diffusion model.
=========================================================================================
Bridges the timeseries VAE and the original (non-timeseries) DDPM by:

1. Loading NeuralVAE from checkpoint.
2. Loading the original U-Net + VQVAE from the static-conditioning diffusion checkpoint.
3. For each test sample:
   - Loads the raw (1, N, T=15) timeseries from the timeseries HDF5.
   - VAE encode → decode → reconstructed (1, N, 15).
   - Mean-pool across T → (1, 1, N) static conditioning vector.
   - Run DDPM reverse diffusion with static conditioner.
4. Saves predicted and true images.

This lets you compare VAE-compressed-then-averaged neural signal vs the raw
trial-averaged signal that the original DDPM was trained on.

Usage:
    python eval_vae_original_ddpm.py \
        --vae_config        configs/ventral_vae_z128_wide_nb2.yaml \
        --diffusion_config  ../reconstruction/configs/ventral_stream_diffusion.yaml
"""

import argparse
import os
import pathlib
import sys

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(_here, '..', 'timeseries'))
sys.path.insert(0, os.path.join(_here, '..', 'reconstruction'))
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import load_config, set_seed

from dataloader_ts import get_timeseries_stimulus_datasets

sys.path.insert(0, os.path.join(_here, 'models'))
from neural_vae import NeuralVAE, NeuralVAEDeep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def resolve(path, config_dir):
    """Resolve a path relative to the config file's directory if not absolute."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        return str(config_dir / p)
    return path


def run_sampling(
    model, vae_neural, scheduler, vqvae,
    train_config, diffusion_model_config, autoencoder_model_config,
    diffusion_config, test_dataset,
    output_dir,
):
    context_dim = (
        diffusion_model_config['condition_config']['neural_condition_config']['neural_embed_dim']
    )
    empty_cond = torch.zeros((1, 1, context_dim), device=device)

    for idx, (test_img, label) in enumerate(test_dataset):
        latent_size = (
            256 // 2 ** sum(autoencoder_model_config['down_sample'])
        )

        xt = torch.randn((
            1,
            autoencoder_model_config['z_channels'],
            latent_size,
            latent_size,
        )).to(device)

        # label: (T=15, N) from timeseries dataset
        neural_raw = label.unsqueeze(0).to(device)          # (1, 15, N)

        # ---- VAE encode → decode ----
        neural_t = neural_raw.transpose(1, 2)               # (1, N, 15)
        with torch.no_grad():
            mu, logvar = vae_neural.encode(neural_t)
            z          = vae_neural.reparameterize(mu, logvar)
            recon_t    = vae_neural.decode(z, T_win=neural_t.shape[2])  # (1, N, 15)

        # Mean-pool over T to get static 315-dim conditioning vector
        neural_mean = recon_t.mean(dim=2)                   # (1, N)
        cond_input  = neural_mean.unsqueeze(1)              # (1, 1, N) = (1, 1, 315)
        uncond_input = empty_cond

        cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)

        # ---- DDPM sampling loop ----
        for i in tqdm(reversed(range(diffusion_config['num_timesteps'])),
                      desc=f'Sample {idx}', leave=False):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = (
                    noise_pred_uncond
                    + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                )
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(
                xt, noise_pred, torch.as_tensor(i).to(device)
            )

            if i == 0:
                ims = vqvae.decode(xt)
            else:
                ims = x0_pred

        ims      = torch.clamp(ims, -1., 1.).detach().cpu()
        ims      = (ims + 1) / 2
        test_img = (test_img + 1) / 2
        ims      = ims.squeeze()

        pred_pil = torchvision.transforms.ToPILImage()(ims)
        pred_pil.thumbnail(
            (test_img.shape[1], test_img.shape[2]), Image.Resampling.LANCZOS
        )
        pred_pil.save(os.path.join(output_dir, f'{idx}_pred.png'))

        true_pil = torchvision.transforms.ToPILImage()(test_img)
        true_pil.save(os.path.join(output_dir, f'{idx}_true.png'))

        pred_pil.close()
        true_pil.close()


def main():
    parser = argparse.ArgumentParser(
        description='Sample from original static DDPM using VAE-decoded + time-averaged neural conditioning'
    )
    parser.add_argument('--vae_config',       required=True, type=str,
                        help='Path to VAE YAML config')
    parser.add_argument('--diffusion_config', required=True, type=str,
                        help='Path to original (static) diffusion YAML config')
    parser.add_argument('--run_id',           default='', type=str,
                        help='Optional tag appended to output directory to avoid overwriting')
    args = parser.parse_args()

    vae_config  = load_config(args.vae_config)
    diff_config = load_config(args.diffusion_config)
    diff_dir    = pathlib.Path(args.diffusion_config).resolve().parent

    set_seed()

    dataset_cfg      = diff_config['dataset_params']
    diffusion_params = diff_config['diffusion_params']
    ldm_params       = diff_config['ldm_params']
    autoenc_params   = diff_config['autoencoder_params']
    train_params     = diff_config['train_params']

    # ---- noise scheduler ----
    scheduler = LinearNoiseScheduler(
        num_timesteps = diffusion_params['num_timesteps'],
        beta_start    = diffusion_params['beta_start'],
        beta_end      = diffusion_params['beta_end'],
    )

    # ---- U-Net ----
    model = Unet(
        im_channels  = autoenc_params['z_channels'],
        model_config = ldm_params,
    ).to(device)
    model.eval()

    ldm_ckpt = resolve(
        os.path.join(train_params['ckpt_dir'], train_params['ldm_ckpt_name']),
        diff_dir,
    )
    if not os.path.exists(ldm_ckpt):
        raise FileNotFoundError(f'Diffusion checkpoint not found: {ldm_ckpt}')
    model.load_state_dict(torch.load(ldm_ckpt, map_location=device))

    # ---- VQVAE ----
    vqvae = VQVAE(
        im_channels  = dataset_cfg['im_channels'],
        model_config = autoenc_params,
    ).to(device)
    vqvae.eval()
    vqvae_ckpt = resolve(
        os.path.join(train_params['ckpt_dir'], train_params['vqvae_autoencoder_ckpt_name']),
        diff_dir,
    )
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))

    # ---- NeuralVAE ----
    vae_dataset_cfg = vae_config['dataset_params']
    vae_model_cfg   = vae_config['vae_params']
    vae_train_cfg   = vae_config['train_params']

    vae_cls = NeuralVAEDeep if vae_model_cfg.get('model_class') == 'deep' else NeuralVAE
    vae_neural = vae_cls(
        num_neurons    = vae_dataset_cfg['num_neurons'],
        enc_channels   = vae_model_cfg['enc_channels'],
        z_channels     = vae_model_cfg['z_channels'],
        kernel_size    = vae_model_cfg.get('kernel_size', 3),
        num_groups     = vae_model_cfg.get('num_groups', 8),
        num_res_blocks = vae_model_cfg.get('num_res_blocks', 1),
    ).to(device)
    vae_neural.eval()

    vae_ckpt_path = os.path.join(vae_train_cfg['ckpt_dir'], 'vae_ckpt_best.pth')
    if not os.path.exists(vae_ckpt_path):
        raise FileNotFoundError(f'VAE checkpoint not found: {vae_ckpt_path}')
    vae_neural.load_state_dict(torch.load(vae_ckpt_path, map_location=device)['vae'])

    # ---- output directory ----
    vae_model_name = vae_config.get('model_name', 'vae')
    suffix = f'_{args.run_id}' if args.run_id else ''
    output_dir = resolve(
        os.path.join(train_params['output_dir'], f'vae_mean_conditioned_{vae_model_name}{suffix}'),
        diff_dir,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving images to: {output_dir}')

    # ---- timeseries test dataset (for VAE input) ----
    num_neurons = vae_dataset_cfg['num_neurons']
    _, test_dataset = get_timeseries_stimulus_datasets(
        name               = diff_config['name'],
        stimulus_size      = 150,
        num_neurons        = num_neurons,
        timeseries_h5_path = vae_dataset_cfg['timeseries_h5_path'],
        modality           = diff_config.get('modality', 'image'),
    )

    with torch.no_grad():
        run_sampling(
            model, vae_neural, scheduler, vqvae,
            train_params, ldm_params, autoenc_params,
            diffusion_params, test_dataset,
            output_dir,
        )

    print('Done.')


if __name__ == '__main__':
    main()
