import h5py

f1_path = '/scratch/groups/anishm/tvsd/monkeyF_things_imgs.mat'
f2_path = '/scratch/groups/anishm/tvsd/monkeyN_things_imgs.mat'

def print_h5_file_summary(file_path, name):
    print(f'=== {name} file ===')
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        print(f'Keys: {keys}')
        for key in keys:
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f'  {key}: Dataset with shape={item.shape}, dtype={item.dtype}')
            elif isinstance(item, h5py.Group):
                # Instead of printing all keys, just summarize
                print(f'  {key}: Group with {len(item.keys())} sub-keys')

print_h5_file_summary(f1_path, 'MonkeyF')
print_h5_file_summary(f2_path, 'MonkeyN')

print('\n=== Checking if train/test counts match ===')
with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
    for split in ['train_imgs', 'test_imgs']:
        if split in f1 and split in f2:
            # Each of these is a Group, so summarize sub-keys
            f1_group = f1[split]
            f2_group = f2[split]
            print(f'{split}: MonkeyF has {len(f1_group["class"])} items, MonkeyN has {len(f2_group["class"])} items')