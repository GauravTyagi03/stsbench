"""
Timeseries Dataloader

Mirrors reconstruction/dataloader.py, but loads binned time-series neural data
from the preprocessed HDF5 (created by preprocess_timeseries.py) instead of
the time-averaged activity stored in the pickle.

The returned datasets yield labels of shape (T, N) per trial instead of (N,).
"""

import pickle

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from os.path import join
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from dataset_ts import TimeseriesImageDataset


def center_crop(img):
    return v2.functional.crop(img, top=115, left=260, height=150, width=150)


def get_transform(name, stimulus_size):
    mean = 0.5
    std  = 0.5
    if 'dorsal' in name.lower():
        transform = T.Compose([
            T.Lambda(center_crop),
            T.Resize(stimulus_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    elif 'ventral' in name.lower():
        transform = T.Compose([
            T.Resize(stimulus_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        raise ValueError(f"Name '{name}' not supported for transforms.")
    return transform


def get_timeseries_stimulus_datasets(
    name,
    stimulus_size=224,
    transform=None,
    num_neurons=2244,
    max_num_dorsal_neurons=2244,
    timeseries_h5_path=None,
    modality='image',
):
    """
    Load image stimulus datasets with binned timeseries neural conditioning labels.

    The preprocessed HDF5 (from preprocess_timeseries.py) must contain:
        'train_timeseries'  shape: (n_train, T, n_electrodes)  float32
        'test_timeseries'   shape: (n_test,  T, n_electrodes)  float32

    Args:
        name:                 Dataset name ('dorsal_stream' or 'ventral_stream').
        stimulus_size:        Target image resize dimension.
        num_neurons:          Number of electrodes to include (used for electrode
                              selection via inc_uids, same logic as original dataloader).
        max_num_dorsal_neurons: Upper bound for dorsal electrode subsampling.
        timeseries_h5_path:   Path to preprocessed HDF5 (required).
        modality:             Currently only 'image' is supported.

    Returns:
        train_dataset, test_dataset  (TimeseriesImageDataset)
    """
    if timeseries_h5_path is None:
        raise ValueError("timeseries_h5_path must be specified in the config under dataset_params")

    with open(f'../dataset/{name}_dataset.pickle', 'rb') as f:
        data = pickle.load(f)

    if 'dorsal' in name.lower():
        with open(f'../dataset/{name}_neuron_table.pickle', 'rb') as f:
            neuron_table = pickle.load(f)

        inc_uids = (
            neuron_table.loc[
                (neuron_table['monkey_id'] == 'T') | (neuron_table['monkey_id'] == 'A')
            ]['neuron_uid']
        ).to_numpy(dtype=np.int64)

        if num_neurons < max_num_dorsal_neurons:
            np.random.seed(42)
            inc_uids = np.random.choice(inc_uids, size=num_neurons, replace=False)

        print(f"Including {len(inc_uids)} neurons from monkey T and A...")
        print(f"Max inc_uid: {np.max(inc_uids)}, max all uid: {neuron_table['neuron_uid'].max()}")

    elif 'ventral' in name.lower():
        actual_num_neurons = data['train_activity'].shape[1]
        print(f"Ventral stream: {actual_num_neurons} neurons in pickle")

        if actual_num_neurons == 0:
            raise ValueError("Ventral pickle has 0 neurons — check preprocessing.")

        inc_uids = np.arange(actual_num_neurons)

        if num_neurons > actual_num_neurons:
            raise ValueError(
                f"Config requests {num_neurons} neurons but data has {actual_num_neurons}. "
                f"Set num_neurons: {actual_num_neurons} in the config."
            )
        if num_neurons < actual_num_neurons:
            print(f"Subsampling from {actual_num_neurons} to {num_neurons} neurons")
            np.random.seed(42)
            inc_uids = np.random.choice(inc_uids, size=num_neurons, replace=False)
    else:
        raise ValueError(f"Name '{name}' not supported for dataset loading.")

    # Load timeseries from preprocessed HDF5, selecting electrodes via inc_uids
    with h5py.File(timeseries_h5_path, 'r') as f:
        n_electrodes_in_file = f['train_timeseries'].shape[2]
        if np.max(inc_uids) >= n_electrodes_in_file:
            raise IndexError(
                f"inc_uids max index {np.max(inc_uids)} exceeds HDF5 electrode dim "
                f"{n_electrodes_in_file}. Check that the HDF5 was built from the same data."
            )
        # Load full arrays first then numpy-index; h5py fancy indexing via point
        # selection is extremely slow on network filesystems (one read per index).
        train_activity = f['train_timeseries'][()][:, :, inc_uids]
        test_activity  = f['test_timeseries'][()][:, :, inc_uids]

    print(f"Loaded train timeseries: {train_activity.shape}")
    print(f"Loaded test timeseries:  {test_activity.shape}")

    directory = join("../dataset/", name)

    if transform is None:
        transform = get_transform(name, stimulus_size)

    train_dataset = TimeseriesImageDataset(
        data['train_stimuli'], train_activity, directory, transform=transform
    )
    test_dataset = TimeseriesImageDataset(
        data['test_stimuli'], test_activity, directory, transform=transform
    )

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=32, val_prop=0, seed=42, num_workers=2):
    generator = torch.Generator().manual_seed(seed)

    if val_prop > 0:
        train_dataset, val_dataset = random_split(
            train_dataset, [1 - val_prop, val_prop], generator=generator
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    if val_prop > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        val_loader = None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
