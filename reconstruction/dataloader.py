import torchvision.transforms as T
import torch 
import pickle
import numpy as np
import pandas as pd

from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split

from dataset import ImageDataset, VideoDataset

from os.path import join

def center_crop(img):
    return v2.functional.crop(img, top=115, left=260, height=150, width=150)
    
def get_transform(name, stimulus_size):
    # normalizes between -1 and 1
    mean = 0.5 
    std = 0.5 

    if 'dorsal' in name.lower(): 
        transform = T.Compose([T.Lambda(center_crop),  # crop every video to 135 x 135
                    T.Resize(stimulus_size),
                    T.ToTensor(),
                    T.Normalize(mean=mean,
                                std=std)])
        
    elif 'ventral' in name.lower():
        transform = T.Compose([T.Resize(stimulus_size), # no crop
                                T.ToTensor(),
                                T.Normalize(mean=mean,
                                            std=std)]) 
    else:
        raise ValueError(f"Name {name} not supported for transforms.")   

    return transform 

def get_stimulus_datasets(name, stimulus_size=224, transform=None, num_neurons=2244, max_num_dorsal_neurons=2244, modality='image'):
    # note that num_neurons parameter here is just to do dorsal stream ablations for revisions 
    with open(f'../dataset/{name}_dataset.pickle', 'rb') as f:
        data = pickle.load(f)

    if 'dorsal' in name.lower():
        # remove neurons from sessions with monkey 'H' if included because their receptive fields are not within this reconstruction crop we are using
        with open(f'../dataset/{name}_neuron_table.pickle', 'rb') as f:
            neuron_table = pickle.load(f)

        inc_uids = (neuron_table.loc[(neuron_table['monkey_id'] == 'T') | (neuron_table['monkey_id'] == 'A')]['neuron_uid']).to_numpy(dtype=np.int64)

        if num_neurons < max_num_dorsal_neurons:
            np.random.seed(42) # seed numpy for reproducibility
            inc_uids = np.random.choice(inc_uids, size=num_neurons, replace=False)

        # sanity checks
        print(f"including {len(inc_uids)} neurons from monkey T and A...")
        print(f"max inc uid {np.max(inc_uids)}, max all uid {neuron_table['neuron_uid'].max()}")
        
        train_activity = data['train_activity'][:, inc_uids]
        test_activity = data['test_activity'][:, inc_uids]
        #train_activity = data['train_activity']
        #test_activity = data['test_activity']
    elif 'ventral' in name.lower():
        train_activity = data['train_activity']
        test_activity = data['test_activity']
    else:
        raise ValueError(f"Name {name} not supported for dataset loading.")   
        
    directory = join("../dataset/", name)
    
    if transform is None:
        transform = get_transform(name, stimulus_size)
        
    if modality == 'video':
        train_dataset = VideoDataset(data['train_stimuli'], train_activity, directory, transform=transform)
        test_dataset = VideoDataset(data['test_stimuli'], test_activity, directory, transform=transform)
    else:
        train_dataset = ImageDataset(data['train_stimuli'], train_activity, directory, transform=transform)
        test_dataset = ImageDataset(data['test_stimuli'], test_activity, directory, transform=transform)

    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, batch_size=32, val_prop=0.05, seed=42, num_workers=2):
    generator = torch.Generator().manual_seed(seed)

    if val_prop > 0:
        train_dataset, val_dataset = random_split(train_dataset, [1-val_prop, val_prop], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    if val_prop > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader