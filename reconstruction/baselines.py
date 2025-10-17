import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from tqdm import tqdm

import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import torchvision
from PIL import Image
import pickle
from dataloader import get_stimulus_datasets, get_dataloaders
from utils import load_config
from utils import set_seed
from pathlib import Path
from copy import deepcopy
    
class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = 2*torch.sigmoid(x)-1
        x = x.view(x.size(0), *self.output_shape)
        return x
        
class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_shape, hidden_dim=32):
        super().__init__()
        
        self.latent_dim = hidden_dim 
        self.output_shape = output_shape 

        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 128 * 8 * 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),   # 32x32 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_shape[0], kernel_size=3, padding=1), # 32x32 -> 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = 2*self.decoder(x)-1
        return x
        
# for lpips zeroing grad
def zero_grads(net):
    for param in net.parameters():
        if param.grad is not None:
            param.grad = None
    
def train_decoder(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device='cuda', lpips_weight=0, weight_decay=0, save_path = None, log_path = None, lr_decay_step=50, lr_decay_gamma=0.1, patience=5, early_stopping=False):
    # clear log
    if log_path is not None: 
        if os.path.exists(log_path):
            os.remove(log_path)
            
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma)
    
    criterion = nn.MSELoss()
    
    if lpips_weight > 0:
        lpips_model = lpips.LPIPS(net='vgg', pnet_tune=True).to(device)  # could also use 'alex'
        zero_grads(lpips_model)
        
    model.train()
    early_stop_cnt = 0
    min_val_loss = np.inf
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            neural_activity = labels.to(device) 

            preds = model(neural_activity)
            
            lpips_loss = 0
            if lpips_weight > 0:
                lpips_loss = torch.mean(lpips_model(preds, images))
            else:
                lpips_loss = 0
                
            loss = criterion(preds, images) + lpips_weight * lpips_loss

            if lpips_weight > 0:
                zero_grads(lpips_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        val_total_loss = 0
        val_mse_loss = 0
        val_lpips_loss = 0
        for images, labels in tqdm(val_loader):
            with torch.no_grad():
                images = images.to(device)
                neural_activity = labels.to(device)
                preds = model(neural_activity)
                
                lpips_loss = torch.tensor(0)
                if lpips_weight > 0:
                    lpips_loss = torch.mean(lpips_model(preds, images))
                loss = criterion(preds, images) + lpips_weight * lpips_loss
                val_total_loss += loss.item()
                val_mse_loss +=  criterion(preds, images).item()
                val_lpips_loss += (lpips_weight * lpips_loss).item()

        avg_val_mse = val_mse_loss/len(val_loader)
        avg_val_lpips = val_lpips_loss/len(val_loader)
        avg_val_loss = val_total_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}  - Avg Val Loss: {avg_val_loss:.4f} - Avg Val LPIPs: {avg_val_lpips:.4f} - Avg Val MSE: {avg_val_mse:.4f}")

        if log_path is not None: 
            with open(log_path, 'a') as f:
                f.write(
                    'Finished epoch: {} | Train Loss : {:.4f} '
                    'Val Loss : {:.4f} | Val Loss LPIPS : {:.4f} | Val Loss MSE : {:.4f}\n'.format(
                        epoch + 1,
                        avg_loss,
                        avg_val_loss,
                        avg_val_lpips,
                        avg_val_mse))

        if early_stopping:
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                cached_model = deepcopy(model)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
    
            if early_stop_cnt >= patience:
                break
                
    if early_stop_cnt >= patience:
        model = cached_model

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser(description='Arguments for baseline training')
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    name = config["name"]
    ckpt_dir = config['ckpt_dir']
    model_name = config.get("model_name", 'cnn')
    stimulus_size = config.get("stimulus_size", 128)
    lpips_weight = config.get("lpips_weight", 0.0)
    batch_size = config.get("batch_size", 64)
    num_epochs = config.get("num_epochs", 25)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0)
    num_neurons = config.get("num_neurons", 315)
    neuron_ablation = config.get("neuron_ablation", False)

    lr_decay_step = config.get("lr_decay_step", 50)
    lr_decay_gamma = config.get("lr_decay_gamma", 0.1)

    patience = config.get("patience", 5)
    early_stopping = config.get("early_stopping", True)

    train_dataset, test_dataset = get_stimulus_datasets(name, stimulus_size=stimulus_size, num_neurons=num_neurons)      
    _, large_test_dataset = get_stimulus_datasets(name, stimulus_size=150, num_neurons=num_neurons)  
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, val_prop=0.05, seed=42)

    set_seed()

    output_shape = (3, stimulus_size, stimulus_size)
    input_dim = train_dataset.labels.shape[1] # number of neurons
    
    if model_name == 'cnn':
        model = CNNDecoder(input_dim, output_shape)
    elif model_name == 'linear':
        model = LinearDecoder(input_dim, output_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make output directories if they do not exist
    if not os.path.exists(config['ckpt_dir']):
        os.mkdir(config['ckpt_dir'])

    if neuron_ablation:
        output_dir = f"./logs/{name}/{model_name}_neuron_ablation/"
    else:
        output_dir = f"./logs/{name}/{model_name}/"
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)        
        
    # 1. train decoder
    if neuron_ablation:
        save_path = Path(os.path.join(ckpt_dir,'model_neuron_ablation.pth'))
    else:
        save_path = Path(os.path.join(ckpt_dir,'model.pth'))
    log_path =  Path(os.path.join(output_dir, f'{model_name}_train.txt'))

    train_decoder(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, lpips_weight=lpips_weight, weight_decay=weight_decay, save_path = save_path, log_path = log_path, lr_decay_step = lr_decay_step, lr_decay_gamma = lr_decay_gamma, early_stopping=early_stopping, patience=patience, device=device)

    # 2. save pred image and true image in output folder 
    os.makedirs(f"./logs/{name}/", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for test_idx, (test_image, test_label) in enumerate((large_test_dataset)):        
        test_im_tensor = large_test_dataset[test_idx][0]
        pred_im_tensor = model(test_label.unsqueeze(0).to(device)).squeeze().cpu()

        test_im = torchvision.transforms.ToPILImage()((test_im_tensor + 1)/2) # undo normalization
        pred_im = torchvision.transforms.ToPILImage()((pred_im_tensor + 1)/2) # undo normalization

        pred_im.save(os.path.join(output_dir, f'{test_idx}_pred.png'))
        test_im.save(os.path.join(output_dir, f'{test_idx}_true.png'))

        pred_im.close()
        test_im.close()
        
    ## 4. all done!
    print(f'Done Training & Testing Baseline {model_name}...')

if __name__ == '__main__':
    main()