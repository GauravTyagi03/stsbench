"""
TimeseriesImageDataset

Mirrors reconstruction/dataset.py ImageDataset, but labels carry an extra
temporal dimension: shape (T, N) instead of (N,).

__getitem__ returns:
    image:  (C, H, W)  — transformed image frame (same as ImageDataset)
    label:  (T, N)     — binned neural time-series for this trial
"""

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from os.path import join
import numpy as np


class TimeseriesImageDataset(Dataset):
    def __init__(self, paths, labels, directory, transform=None, extension='mp4'):
        """
        Args:
            paths:      List of stimulus video filenames (no extension).
            labels:     np.ndarray of shape (n_trials, T, N).
            directory:  Root directory containing the stimulus videos.
            transform:  Torchvision transform applied to each image frame.
            extension:  Video file extension (default 'mp4').
        """
        self.video_dir = directory
        self.image_paths = paths
        # labels shape: (n_trials, T, N)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.labels[torch.isnan(self.labels)] = 0
        self.transform = transform if transform else transforms.ToTensor()
        self.extension = extension

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(join(self.video_dir, self.image_paths[idx] + '.' + self.extension))
        _, image = cap.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        cap.release()
        label = self.labels[idx]  # (T, N)
        return image, label
