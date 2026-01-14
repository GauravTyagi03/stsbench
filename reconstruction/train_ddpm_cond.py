import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch 
import os
from dataloader import get_stimulus_datasets, get_dataloaders

from models.unet_cond_base import Unet
from models.vqvae import VQVAE

from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from utils import load_config, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    ########## Read the config file #############
    parser = argparse.ArgumentParser(description='Arguments for vqvae training')
    parser.add_argument('--config', default='configs/dorsal_stream_diffusion.yaml', type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)    

    name = config['name']
    modality = config.get("modality", "image")

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    set_seed()
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ########## Instantiate condition components #############
    empty_neural_embed = None
    condition_types = []
    condition_config = diffusion_model_config['condition_config']
    if condition_config is not None:
        condition_types = condition_config['condition_types']
        if 'neural' in condition_types:
            with torch.no_grad():
                context_dim = condition_config['neural_condition_config']['neural_embed_dim']
                empty_neural_embed = torch.zeros((1, context_dim), device=device)

    ########## Instantiate dataset and loader #############
    train_dataset, test_dataset = get_stimulus_datasets(name, stimulus_size=dataset_config['im_size'], num_neurons=condition_config['neural_condition_config']['neural_embed_dim'], modality=modality)        
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=train_config['ldm_batch_size'], val_prop=0, seed=42, num_workers=8)
             
    
    ########## Instantiate Unet model #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    ########## Load pretrained VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae.load_state_dict(torch.load(os.path.join(train_config['ckpt_dir'], train_config['vqvae_autoencoder_ckpt_name']), map_location=device))

    # freeze params
    for param in vae.parameters():
        param.requires_grad = False

    ########## Setup Logging #############
    log_file = os.path.join(train_config['output_dir'], train_config['ldm_log_name'])
    if log_file is not None: 
        if os.path.exists(log_file):
            os.remove(log_file)
            
    ########## Training #############
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    for epoch_idx in range(num_epochs):
        losses = []
        for im, cond_input in tqdm(train_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            with torch.no_grad():
                im, _ = vae.encode(im)
            ########### Handling Conditional Input ###########
            if 'neural' in condition_types:
                with torch.no_grad():
                    # need to unsqueeze to add dummy sequence dim + drop some percent of class labels 
                    neural_condition = cond_input.unsqueeze(1).to(device) # the unsqueeze 1 is to create dummy sequence dim
                    neural_drop_prob = condition_config['neural_condition_config']['cond_drop_prob']
                    if neural_drop_prob > 0:
                        neural_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0, 1) < neural_drop_prob
                        neural_condition[neural_drop_mask, :, :] = empty_neural_embed.unsqueeze(0)

                    cond_input = neural_condition
            ################################################
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1, np.mean(losses)))
        
        with open(log_file, 'a') as f:
            f.write('Finished epoch:{} | Loss : {:.4f}\n'.format(
                epoch_idx + 1,
                np.mean(losses)))
            
        if epoch_idx % 5 == 0 or epoch_idx == num_epochs - 1:    
            torch.save(model.state_dict(), os.path.join(train_config['ckpt_dir'], train_config['ldm_ckpt_name']))
        
    print('Done Training ...')
    
if __name__ == '__main__':
    main()
