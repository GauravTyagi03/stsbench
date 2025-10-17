import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import numpy as np
from PIL import Image, ImageFilter
import torch
from utils import load_config, set_seed
from dataloader import get_stimulus_datasets, get_dataloaders


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, test_dataset, modality='image'):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    # sample for each stimulus in the text set
    for idx, (test_img, label) in enumerate(test_dataset):    
        latent_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
        
        ########### Sample random noise latent ##########
        xt = torch.randn((1,
                            autoencoder_model_config['z_channels'],
                            latent_size,
                            latent_size)).to(device)
        ###############################################
        
        ############ Create Conditional input ###############
        neural_prompt = label.unsqueeze(0).unsqueeze(0) # add a batch dimension & sequence dim
        neural_prompt = neural_prompt.to(device)
        
        context_dim = diffusion_model_config['condition_config']['neural_condition_config']['neural_embed_dim']
        empty_neural_embed = torch.zeros((1, 1, context_dim), device=device)

        assert empty_neural_embed.shape == neural_prompt.shape

        uncond_input = empty_neural_embed
        cond_input = neural_prompt
        
        ###############################################
        
        # By default classifier free guidance is disabled
        # Change value in config or change default value here to enable it
        cf_guidance_scale = 1.0 

        ################# Sampling Loop ########################
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)
            
            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
            if i == 0:
                # Decode ONLY the final image to save time
                ims = vae.decode(xt)
            else:
                ims = x0_pred
            
        ims = torch.clamp(ims, -1., 1.).detach().cpu() # clamp pred and move to cpu
        
        ims = (ims + 1) / 2 # undo normalization
        test_img = (test_img + 1)/2 # undo normalization

        ims = ims.squeeze()
        
        if modality == 'video':
            for im_id in range(ims.shape[0]):
                pred_img = torchvision.transforms.ToPILImage()(ims[im_id])
                pred_img.thumbnail((test_img.shape[1], test_img.shape[2]), Image.Resampling.LANCZOS) # resize to starting image size
                pred_img.save(os.path.join(train_config['output_dir'], f'{idx}_{im_id}_pred.png'))
        
                test_im = torchvision.transforms.ToPILImage()(test_img[im_id])
                test_im.save(os.path.join(train_config['output_dir'], f'{idx}_{im_id}_true.png'))
                pred_img.close()
                test_im.close()
        else:
            pred_img = torchvision.transforms.ToPILImage()(ims)
            pred_img.thumbnail((test_img.shape[1], test_img.shape[2]), Image.Resampling.LANCZOS) # resize to starting image size
            pred_img.save(os.path.join(train_config['output_dir'], f'{idx}_pred.png'))
    
            test_img = torchvision.transforms.ToPILImage()(test_img)
            test_img.save(os.path.join(train_config['output_dir'], f'{idx}_true.png'))
            pred_img.close()
            test_img.close()
        ##############################################################
    

def main():
    ########## Read the config file #############
    parser = argparse.ArgumentParser(description='Arguments for sampling')
    parser.add_argument('--config', default='configs/dorsal_stream_diffusion.yaml', type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)    
    ########################
    modality = config.get("modality", "image")

    name = config['name']
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    condition_config = diffusion_model_config['condition_config']

    set_seed()
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()

    ldm_ckpt_path = os.path.join(train_config['ckpt_dir'], train_config['ldm_ckpt_name'])

    if os.path.exists(ldm_ckpt_path):
        model.load_state_dict(torch.load(ldm_ckpt_path, map_location=device))
    else:
        raise Exception('Model checkpoint {} not found'.format(ldm_ckpt_path))
    #####################################
    
    # Create output directories if they do not exist
    if not os.path.exists(train_config['output_dir']):
        os.mkdir(train_config['output_dir'])
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae.load_state_dict(torch.load(os.path.join(train_config['ckpt_dir'], train_config['vqvae_autoencoder_ckpt_name']), map_location=device))
    #####################################
    
    _, test_dataset = get_stimulus_datasets(name, stimulus_size=150, num_neurons=condition_config['neural_condition_config']['neural_embed_dim'], modality=modality) # warning - note that this is hardcoded here, be careful when changing code            

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, test_dataset, modality)


if __name__ == '__main__':
    main()
