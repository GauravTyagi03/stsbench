"""
SlidingWindowNeuralDataset
==========================
Wraps ventral_stream_timeseries_preprocessed.h5 and enumerates overlapping
sliding windows over the T=15 time axis.

__getitem__ returns: (N, T_win) tensor  — N=315 channels, T_win time steps
  (no image; the VAE trains on neural data alone)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowNeuralDataset(Dataset):
    """
    Parameters
    ----------
    h5_path : str
        Path to the preprocessed HDF5 file.
    split : str
        'train' or 'test' — reads the corresponding key from the HDF5.
    T_win : int
        Sliding window size in bins.
    win_stride : int
        Stride between windows (training only; test uses a single window).
    use_sliding : bool
        If True, enumerate overlapping windows; if False, use a single window
        starting at bin 0 (pad or truncate to T_win if needed).
    """

    def __init__(self, h5_path, split='train', T_win=10, win_stride=3,
                 use_sliding=True):
        super().__init__()
        self.T_win  = T_win
        self.use_sliding = use_sliding

        key = 'train_timeseries' if split == 'train' else 'test_timeseries'
        with h5py.File(h5_path, 'r') as f:
            data = f[key][:]  # (n_trials, T=15, N=315)

        # Replace NaNs with zero (mirrors TimeseriesImageDataset)
        data = np.where(np.isnan(data), 0.0, data)
        self.data = torch.tensor(data, dtype=torch.float32)  # (n, T, N)

        n_trials, T_full, _ = self.data.shape

        if use_sliding:
            # enumerate (sample_idx, win_start) pairs
            starts = list(range(0, T_full - T_win + 1, win_stride))
            self.indices = [(i, s) for i in range(n_trials) for s in starts]
        else:
            # single window per sample; clamp to available length
            self.indices = [(i, 0) for i in range(n_trials)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, win_start = self.indices[idx]
        win = self.data[sample_idx, win_start : win_start + self.T_win, :]  # (T_win, N)
        return win.transpose(0, 1)  # (N, T_win)
