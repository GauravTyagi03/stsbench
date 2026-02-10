# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import h5py
import pickle
import os
from scipy.io import loadmat


def load_mat_file(filepath):
    """
    Load .mat file with automatic format detection.

    Tries HDF5/MATLAB v7.3 format first (h5py), falls back to
    MATLAB v5 format (scipy.io.loadmat) if needed.

    Returns:
        dict: Data loaded from the .mat file
    """
    try:
        # Try HDF5 format first (MATLAB v7.3)
        with h5py.File(filepath, "r") as f:
            data = {key: np.array(f[key]) for key in f.keys() if not key.startswith('#')}
        return data
    except (OSError, IOError):
        # Fall back to MATLAB v5 format
        print(f"  Note: Loading {os.path.basename(filepath)} with scipy (MATLAB v5 format)")
        return loadmat(filepath)

# paths
#normMUA_paths = ['/scratch/groups/anishm/tvsd/monkeyF_THINGS_normMUA.mat', '/scratch/groups/anishm/tvsd/monkeyN_THINGS_normMUA.mat']
normMUA_paths = ['/oak/stanford/groups/anishm/gtyagi/stsbench/results/monkeyN_paper_normalized.mat', '/oak/stanford/groups/anishm/gtyagi/stsbench/results/monkeyF_paper_normalized.mat']
# things path - this should be the path you downloaded and unzipped the things dataset _things_database_images.zip
things_folder = "/scratch/groups/anishm/things_dataset/object_images/"
things_path = "/scratch/groups/anishm/tvsd/monkeyN_things_imgs.mat"

# this loads the MUA data from both monkeys in V4 recordings
v4_range_N = (512, 768) # is monkey N channels for V4
v4_range_F = (832, 1024) # is monkey F channels for V4


all_oracle = []
all_reliab = []
all_train = []
all_test = []

for normMUA_path in normMUA_paths:
    # Load with automatic format detection
    data = load_mat_file(normMUA_path)

    if "monkeyN" in normMUA_path:
        v4_range = v4_range_N
    elif "monkeyF" in normMUA_path:
        v4_range = v4_range_F
    else:
        raise ValueError(0)

    train_MUA = data["train_MUA"][:, v4_range[0]:v4_range[1]]        # shape: (n_train_stimuli, n_electrodes)
    test_MUA = data["test_MUA"][:, v4_range[0]:v4_range[1]]          # shape: (n_test_stimuli, n_electrodes)
    reliab = np.mean(data["reliab"], 0)[v4_range[0]:v4_range[1]]    # shape: (n_electrodes)
    oracle = data["oracle"][v4_range[0]:v4_range[1]]                # shape: (n_electrodes)

    all_oracle.append(oracle)
    all_reliab.append(reliab)
    all_train.append(train_MUA)
    all_test.append(test_MUA)

oracle = np.concatenate(all_oracle)
reliab = np.concatenate(all_reliab)
train_MUA = np.column_stack(all_train)
test_MUA = np.column_stack(all_test)

# %%
# exclude channels with low reliability 
include_index = reliab > 0.3

oracle = oracle[include_index]
reliab = reliab[include_index]
train_activity = train_MUA[:, include_index]
test_activity = test_MUA[:, include_index]

# %%
with h5py.File(things_path, "r") as f:
    def decode_references(dset):
        """Dereference and decode all object references in a dataset."""
        decoded_strings = []
        for i in range(dset.shape[0]):  # Iterate over the first column
            ref = dset[i][0]
            actual_data = f[ref][()]  # Dereference
            decoded_str = actual_data.tobytes().decode("utf-16-le")  # Decode properly
            decoded_str = decoded_str.replace("\\", "/")  # Flip backslashes if needed
            decoded_strings.append(decoded_str)        
        return np.array(decoded_strings)

    # Decode train and test image paths
    print(f["train_imgs"]['things_path'])
    train_imgs = decode_references(f["train_imgs"]['things_path'])
    test_imgs = decode_references(f["test_imgs"]['things_path'])

    print(f"Decoded {len(train_imgs)} train image paths.")
    print(f"Decoded {len(test_imgs)} test image paths.")

for i in range(len(train_imgs)):
    train_imgs[i] = os.path.splitext(os.path.basename(train_imgs[i]))[0]

for i in range(len(test_imgs)):
    test_imgs[i] = os.path.splitext(os.path.basename(test_imgs[i]))[0]

test_imgs = test_imgs.astype(dtype='<U22')
train_imgs = train_imgs.astype(dtype='<U22')

# %%
dataset = dict()
dataset["train_stimuli"] = train_imgs
dataset["test_stimuli"] = test_imgs
dataset["reliab"] = reliab
dataset["test_activity"] = test_activity
dataset["train_activity"] = train_activity

# %%
# Create output directory if it doesn't exist
os.makedirs("../dataset", exist_ok=True)

with open("../dataset/ventral_stream_dataset.pickle", "wb") as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
import cv2
from pathlib import Path

def convert_images_to_videos(things_folder, output_dir = "../dataset/ventral_stream/"):
    os.makedirs(output_dir, exist_ok=True)    

    ventral_images_path = Path(things_folder)

    # Find all image files in the folder and subfolders
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(ventral_images_path.rglob(ext))
    print(ventral_images_path)
    for img_path in image_paths:
        flat_name = img_path.stem + '.mp4'
        output_path = Path(output_dir) / flat_name

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        height, width, _ = image.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, 25, (width, height))

        # write 5 frames of the image
        for _ in range(5):
            video_writer.write(image)

        video_writer.release()
        print(f"Saved video: {output_path}")

convert_images_to_videos(things_folder)
