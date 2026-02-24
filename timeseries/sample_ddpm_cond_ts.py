"""
Sample from Trained Timeseries Conditional DDPM
================================================
Mirrors reconstruction/sample_ddpm_cond.py with the following changes:

1. Loads binned timeseries test labels  (T, N)  via dataloader_ts.py
2. Loads TemporalNeuralConditioner from the combined checkpoint
3. label.unsqueeze(0) -> (1, T, N) (one unsqueeze for batch only; T dim already present)
4. Passes raw neural prompt through temporal_cond to get (1, T, d_model) context
5. Null conditioning is temporal_cond applied to an all-zero (1, T, N) input,
   consistent with how dropout-zeroed samples were conditioned during training
"""

import argparse
import os
import sys

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# ---- shared modules live in reconstruction/ ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reconstruction'))
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import load_config, set_seed

# ---- timeseries-specific modules ----
from dataloader_ts import get_timeseries_stimulus_datasets
from ts_models.temporal_conditioner import TemporalNeuralConditioner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(
    model, temporal_cond, scheduler,
    train_config, diffusion_model_config, autoencoder_model_config,
    diffusion_config, dataset_config, vae, test_dataset,
    num_bins, num_neurons,
):
    r"""
    Sample stepwise by going backward one timestep at a time.
    Saves x0 prediction at the final step.
    """
    for idx, (test_img, label) in enumerate(test_dataset):
        latent_size = (
            dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
        )

        ########### Sample random noise latent ###########
        xt = torch.randn((
            1,
            autoencoder_model_config['z_channels'],
            latent_size,
            latent_size,
        )).to(device)
        ##################################################

        ########### Build conditional and null context ###########
        # label shape: (T, N) — from TimeseriesImageDataset
        # Add batch dim: (T, N) -> (1, T, N)
        neural_prompt_raw = label.unsqueeze(0).to(device)
        cond_input = temporal_cond(neural_prompt_raw)           # (1, T, d_model)

        # Null conditioning: conditioner output for all-zero neural input.
        # During training, dropped samples had their (B, T, N) input zeroed before
        # the conditioner, so this is the exact null token the model learned against.
        null_raw   = torch.zeros(1, num_bins, num_neurons, device=device)
        uncond_input = temporal_cond(null_raw)                  # (1, T, d_model)

        assert cond_input.shape == uncond_input.shape
        ##########################################################

        cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)

        ################# Sampling Loop ######################
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
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
                ims = vae.decode(xt)
            else:
                ims = x0_pred
        ######################################################

        ims      = torch.clamp(ims, -1., 1.).detach().cpu()
        ims      = (ims + 1) / 2
        test_img = (test_img + 1) / 2
        ims      = ims.squeeze()

        pred_img = torchvision.transforms.ToPILImage()(ims)
        pred_img.thumbnail(
            (test_img.shape[1], test_img.shape[2]), Image.Resampling.LANCZOS
        )
        pred_img.save(os.path.join(train_config['output_dir'], f'{idx}_pred.png'))

        test_img_pil = torchvision.transforms.ToPILImage()(test_img)
        test_img_pil.save(os.path.join(train_config['output_dir'], f'{idx}_true.png'))
        pred_img.close()
        test_img_pil.close()


def main():
    parser = argparse.ArgumentParser(description='Arguments for timeseries DDPM sampling')
    parser.add_argument('--config', default='configs/dorsal_stream_diffusion_ts.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    modality                 = config.get('modality', 'image')
    name                     = config['name']
    diffusion_config         = config['diffusion_params']
    dataset_config           = config['dataset_params']
    diffusion_model_config   = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config             = config['train_params']
    condition_config         = diffusion_model_config['condition_config']
    neural_cond_config       = condition_config['neural_condition_config']

    set_seed()

    num_bins         = neural_cond_config['num_bins']
    num_neurons      = neural_cond_config['num_neurons']
    temporal_d_model = neural_cond_config['temporal_d_model']
    temporal_dropout = neural_cond_config.get('temporal_dropout', 0.0)

    # ---- noise scheduler ----
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    # ---- U-Net ----
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config,
    ).to(device)
    model.eval()

    # ---- TemporalNeuralConditioner ----
    temporal_cond = TemporalNeuralConditioner(
        n_neurons=num_neurons,
        d_model=temporal_d_model,
        num_bins=num_bins,
        dropout=temporal_dropout,
    ).to(device)
    temporal_cond.eval()

    # ---- load checkpoint (combined dict with 'model' and 'temporal_cond' keys) ----
    ldm_ckpt_path = os.path.join(train_config['ckpt_dir'], train_config['ldm_ckpt_name'])
    if not os.path.exists(ldm_ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ldm_ckpt_path}')
    ckpt = torch.load(ldm_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    temporal_cond.load_state_dict(ckpt['temporal_cond'])

    os.makedirs(train_config['output_dir'], exist_ok=True)

    # ---- VQ-VAE ----
    vae = VQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config,
    ).to(device)
    vae.eval()
    vqvae_ckpt_dir = train_config.get('vqvae_ckpt_dir', train_config['ckpt_dir'])
    vae.load_state_dict(torch.load(
        os.path.join(vqvae_ckpt_dir, train_config['vqvae_autoencoder_ckpt_name']),
        map_location=device,
    ))

    # ---- test dataset (stimulus_size=150 matches original sample script) ----
    _, test_dataset = get_timeseries_stimulus_datasets(
        name=name,
        stimulus_size=150,  # warning: hardcoded to match original sample_ddpm_cond.py
        num_neurons=num_neurons,
        timeseries_h5_path=dataset_config['timeseries_h5_path'],
        modality=modality,
    )

    with torch.no_grad():
        sample(
            model, temporal_cond, scheduler,
            train_config, diffusion_model_config, autoencoder_model_config,
            diffusion_config, dataset_config, vae, test_dataset,
            num_bins, num_neurons,
        )


if __name__ == '__main__':
    main()
