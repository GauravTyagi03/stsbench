import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm

from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator

from dataloader import get_stimulus_datasets, get_dataloaders
from torch.optim import Adam
from torchvision.utils import make_grid
from utils import load_config
from utils import set_seed
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='Arguments for vqvae training')
    parser.add_argument('--config', default='configs/dorsal_stream_diffusion.yaml', type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)    
    modality = config.get("modality", "image")
    name = config["name"]
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    set_seed()
    
    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)

    train_dataset, test_dataset = get_stimulus_datasets(name, stimulus_size=dataset_config['im_size'], modality=modality)        
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=train_config['autoencoder_batch_size'], val_prop=0, seed=42)
    
    # Create output directories
    os.makedirs(train_config['output_dir'], exist_ok=True)
    os.makedirs(train_config['ckpt_dir'], exist_ok=True)
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    
    # Discriminator Loss 
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0

    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0    

    # set up log and delete if exists 
    log_file = os.path.join(train_config['output_dir'], train_config['vqvae_autoencoder_log_name'])

    if log_file is not None: 
        if os.path.exists(log_file):
            os.remove(log_file)
            
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im, _ in tqdm(train_loader):
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic to Visualize Samples 
            if step_count % image_save_steps == 0:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                if modality == 'video':
                    # Current video shape: (B, T, H, W), where T is time/frames
                    num_frames = save_input.shape[1]
                    # Reshape to (B*T, 1, H, W) to create a batch of single-channel images
                    save_input = save_input.reshape(-1, 1, save_input.shape[2], save_input.shape[3])
                    save_output = save_output.reshape(-1, 1, save_output.shape[2], save_output.shape[3])
                    
                    # Set nrow to number of frames to group each video's frames in one row
                    grid_nrow = num_frames
                else: # Standard image modality: (B, C, H, W)
                    grid_nrow = sample_size
                                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=grid_nrow)
                img = torchvision.transforms.ToPILImage()(grid)
                img_output_path = os.path.join(train_config['output_dir'],'vqvae_autoencoder_samples')
                os.makedirs(img_output_path, exist_ok=True)
                img.save(os.path.join(img_output_path,
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            if modality == 'video':
                lpips_loss = 0.0
                for i in range(im.shape[1]):
                    lpips_im = im[:, i, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
                    lpips_output = output[:, i, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
                    lpips_loss += torch.mean(lpips_model(lpips_output, lpips_im))
                lpips_loss /= im.shape[1]
            else:
                lpips_loss = torch.mean(lpips_model(output, im))
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
    
        with open(log_file, 'a') as f:
            if len(disc_losses) > 0:
                f.write(
                    'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                    'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}\n'.format(
                        epoch_idx + 1,
                        np.mean(recon_losses),
                        np.mean(perceptual_losses),
                        np.mean(codebook_losses),
                        np.mean(gen_losses),
                        np.mean(disc_losses)))
            else:
                f.write(
                    'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}\n'.format(
                        epoch_idx + 1,
                        np.mean(recon_losses),
                        np.mean(perceptual_losses),
                        np.mean(codebook_losses)))
                
        torch.save(model.state_dict(), os.path.join(train_config['ckpt_dir'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['ckpt_dir'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
    print('Done Training...')


if __name__ == '__main__':
    main()
