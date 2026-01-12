import h5py
import numpy as np

f1_path = '/scratch/groups/anishm/tvsd/monkeyF_things_imgs.mat'
f2_path = '/scratch/groups/anishm/tvsd/monkeyN_things_imgs.mat'

print('=== MonkeyF file ===')
with h5py.File(f1_path, 'r') as f:
    print('Keys:', list(f.keys()))
    for key in f.keys():
        print(f'  {key}: shape = {f[key].shape}, dtype = {f[key].dtype}')

print('\n=== MonkeyN file ===')
with h5py.File(f2_path, 'r') as f:
    print('Keys:', list(f.keys()))
    for key in f.keys():
        print(f'  {key}: shape = {f[key].shape}, dtype = {f[key].dtype}')

print('\n=== Checking if train/test counts match ===')
with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
    if 'train_imgs' in f1 and 'train_imgs' in f2:
        print(f'MonkeyF train_imgs shape: {f1['train_imgs'].shape}')
        print(f'MonkeyN train_imgs shape: {f2['train_imgs'].shape}')
    if 'test_imgs' in f1 and 'test_imgs' in f2:
        print(f'MonkeyF test_imgs shape: {f1['test_imgs'].shape}')
        print(f'MonkeyN test_imgs shape: {f2['test_imgs'].shape}')