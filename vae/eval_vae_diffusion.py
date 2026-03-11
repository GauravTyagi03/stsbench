"""
Evaluate VAE-decoded conditioning with existing Conv1d Diffusion model
=======================================================================
1. Loads NeuralVAE from checkpoint.
2. Loads U-Net + TemporalNeuralConditioner from existing diffusion checkpoint.
3. For each test sample:
   - raw (1, N, T_win=15) → VAE encode → decode → reconstructed (1, N, 15)
   - transpose to (1, 15, N) → TemporalNeuralConditioner → DDPM sampling
4. Saves predicted and true images to a new output directory.

Usage:
    python eval_vae_diffusion.py \
        --vae_config    configs/ventral_vae_z64_beta001.yaml \
        --diffusion_config  ../timeseries/configs/ventral_stream_diffusion_ts_conv1d.yaml
"""

import argparse
import os
import sys

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# ensure the vae/ directory itself is on the path (for models/, vae_dataset)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# shared reconstruction utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reconstruction'))
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import load_config, set_seed

# timeseries modules (for TemporalNeuralConditioner + test dataset)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'timeseries'))
from dataloader_ts import get_timeseries_stimulus_datasets
from ts_models.temporal_conditioner import TemporalNeuralConditioner

# VAE
from models.neural_vae import NeuralVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def run_sampling(
    model, temporal_cond, vae_neural, scheduler,
    train_config, diffusion_model_config, autoencoder_model_config,
    diffusion_config, dataset_config, vqvae, test_dataset,
    num_bins, num_neurons, output_dir,
):
    """
    For each test sample:
      1. VAE encode-decode the raw neural signal
      2. Feed reconstructed signal through TemporalNeuralConditioner
      3. Run DDPM reverse diffusion
      4. Decode latent with VQVAE and save image
    """
    for idx, (test_img, label) in enumerate(test_dataset):
        latent_size = (
            dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
        )

        xt = torch.randn((
            1,
            autoencoder_model_config['z_channels'],
            latent_size,
            latent_size,
        )).to(device)

        # label: (T, N) from TimeseriesImageDataset; add batch → (1, T, N)
        neural_raw = label.unsqueeze(0).to(device)   # (1, 15, N)

        # ---- VAE encode → decode ----
        # VAE expects (B, N, T_win); transpose (1, 15, N) → (1, N, 15)
        neural_t = neural_raw.transpose(1, 2)        # (1, N, 15)
        mu, logvar = vae_neural.encode(neural_t)
        z          = vae_neural.reparameterize(mu, logvar)
        recon_t    = vae_neural.decode(z, T_win=neural_t.shape[2])  # (1, N, 15)
        # transpose back to (1, 15, N) for TemporalNeuralConditioner
        neural_recon = recon_t.transpose(1, 2)       # (1, 15, N)

        # ---- conditioner ----
        cond_input   = temporal_cond(neural_recon)                            # (1, T, d_model)
        null_raw     = torch.zeros(1, num_bins, num_neurons, device=device)
        uncond_input = temporal_cond(null_raw)                                # (1, T, d_model)

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
        description='Sample from Conv1d DDPM using VAE-decoded neural conditioning'
    )
    parser.add_argument('--vae_config',       required=True, type=str,
                        help='Path to VAE YAML config')
    parser.add_argument('--diffusion_config', required=True, type=str,
                        help='Path to existing conv1d diffusion YAML config')
    args = parser.parse_args()

    vae_config  = load_config(args.vae_config)
    diff_config = load_config(args.diffusion_config)

    set_seed()

    dataset_cfg        = diff_config['dataset_params']
    diffusion_params   = diff_config['diffusion_params']
    ldm_params         = diff_config['ldm_params']
    autoenc_params     = diff_config['autoencoder_params']
    train_params       = diff_config['train_params']
    cond_cfg           = ldm_params['condition_config']
    neural_cfg         = cond_cfg['neural_condition_config']

    num_bins    = neural_cfg['num_bins']
    num_neurons = neural_cfg['num_neurons']
    temporal_encoder_type = neural_cfg.get('temporal_encoder_type', 'conv1d')
    temporal_d_model      = neural_cfg['temporal_d_model']

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

    # ---- TemporalNeuralConditioner ----
    temporal_cond = TemporalNeuralConditioner(
        n_neurons            = num_neurons,
        d_model              = temporal_d_model,
        num_bins             = num_bins,
        dropout              = neural_cfg.get('temporal_dropout', 0.0),
        temporal_encoder_type= temporal_encoder_type,
        conv_kernel_size     = neural_cfg.get('conv_kernel_size', 3),
        bin_start            = neural_cfg.get('bin_start', 0),
    ).to(device)
    temporal_cond.eval()

    # ---- load diffusion checkpoint ----
    ldm_ckpt = os.path.join(train_params['ckpt_dir'], train_params['ldm_ckpt_name'])
    if not os.path.exists(ldm_ckpt):
        raise FileNotFoundError(f'Diffusion checkpoint not found: {ldm_ckpt}')
    ckpt = torch.load(ldm_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    temporal_cond.load_state_dict(ckpt['temporal_cond'])

    # ---- VQVAE (image decoder) ----
    vqvae = VQVAE(
        im_channels  = dataset_cfg['im_channels'],
        model_config = autoenc_params,
    ).to(device)
    vqvae.eval()
    vqvae_ckpt_dir = train_params.get('vqvae_ckpt_dir', train_params['ckpt_dir'])
    vqvae.load_state_dict(torch.load(
        os.path.join(vqvae_ckpt_dir, train_params['vqvae_autoencoder_ckpt_name']),
        map_location=device,
    ))

    # ---- NeuralVAE ----
    vae_dataset_cfg = vae_config['dataset_params']
    vae_model_cfg   = vae_config['vae_params']
    vae_train_cfg   = vae_config['train_params']

    vae_neural = NeuralVAE(
        num_neurons  = vae_dataset_cfg['num_neurons'],
        enc_channels = vae_model_cfg['enc_channels'],
        z_channels   = vae_model_cfg['z_channels'],
        kernel_size  = vae_model_cfg.get('kernel_size', 3),
        num_groups   = vae_model_cfg.get('num_groups', 8),
    ).to(device)
    vae_neural.eval()

    vae_ckpt_path = os.path.join(vae_train_cfg['ckpt_dir'], vae_train_cfg['ckpt_name'])
    if not os.path.exists(vae_ckpt_path):
        raise FileNotFoundError(f'VAE checkpoint not found: {vae_ckpt_path}')
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_neural.load_state_dict(vae_ckpt['vae'])

    # ---- output directory ----
    # Use a sub-directory of the diffusion output dir, named after the VAE model
    vae_model_name = vae_config.get('model_name', 'vae')
    output_dir = os.path.join(
        train_params['output_dir'], f'vae_conditioned_{vae_model_name}'
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving images to: {output_dir}')

    # ---- test dataset ----
    modality = diff_config.get('modality', 'image')
    name     = diff_config['name']
    _, test_dataset = get_timeseries_stimulus_datasets(
        name             = name,
        stimulus_size    = 150,
        num_neurons      = num_neurons,
        timeseries_h5_path = dataset_cfg['timeseries_h5_path'],
        modality         = modality,
    )

    with torch.no_grad():
        run_sampling(
            model, temporal_cond, vae_neural, scheduler,
            train_params, ldm_params, autoenc_params,
            diffusion_params, dataset_cfg, vqvae, test_dataset,
            num_bins, num_neurons, output_dir,
        )

    print('Done.')


if __name__ == '__main__':
    main()
