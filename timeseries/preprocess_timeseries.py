"""
Preprocess Timeseries Data for DDPM Conditioning
=================================================
Reads the normalized timeseries HDF5 (from normalization/alternative_timeseries_norm.py),
bins individual timepoints into T equal-width bins, and splits into train/test
by averaging over stimulus repetitions using the trial-to-stimulus mapping in
the original raw .mat file.

Input:
  {monkey}_timeseries_normalized.h5   (from normalization/)
      dataset 'timeseries_normalized'  shape: (n_timepoints, n_electrodes, n_trials)
  {monkey}_THINGS_MUA_trials.mat      (raw data)
      ALLMAT[1] — train_stim_id per trial (1-indexed; 0 = not a train trial)
      ALLMAT[2] — test_stim_id  per trial (1-indexed; 0 = not a test trial)

Output:
  {name}_timeseries_preprocessed.h5   with:
      'train_timeseries'  shape: (n_train, T, n_electrodes)  float32
      'test_timeseries'   shape: (n_test,  T, n_electrodes)  float32
  attrs: num_bins, n_train, n_test

Row order matches the pickle (sorted stimulus ID), so image-neural pairings
in the dataloader remain correct.

Usage:
  python preprocess_timeseries.py \\
      --timeseries_h5  /path/to/monkeyN_timeseries_normalized.h5 \\
      --raw_mat        /path/to/monkeyN_THINGS_MUA_trials.mat \\
      --output_path    /path/to/ventral_stream_timeseries_preprocessed.h5 \\
      --num_bins       15
"""

import argparse
import os

import h5py
import numpy as np


def bin_timeseries(data: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Average a timeseries array into num_bins equal-width bins.

    Args:
        data:     (n_timepoints, n_electrodes, n_trials)
        num_bins: target number of bins T

    Returns:
        binned:   (num_bins, n_electrodes, n_trials)
    """
    n_timepoints = data.shape[0]
    bin_width = n_timepoints // num_bins
    usable = bin_width * num_bins

    if usable < n_timepoints:
        print(f"  Truncating {n_timepoints} -> {usable} timepoints to fit {num_bins} x {bin_width} bins")
    data = data[:usable]  # trim to exact multiple

    # Reshape to (num_bins, bin_width, n_electrodes, n_trials) then mean over bin axis
    binned = data.reshape(num_bins, bin_width, data.shape[1], data.shape[2]).mean(axis=1)
    return binned  # (num_bins, n_electrodes, n_trials)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess normalized timeseries HDF5 for DDPM conditioning'
    )
    parser.add_argument('--timeseries_h5', type=str, required=True,
                        help='Path to *_timeseries_normalized.h5 (from normalization/)')
    parser.add_argument('--raw_mat', type=str, required=True,
                        help='Path to {monkey}_THINGS_MUA_trials.mat (for trial-to-stimulus mapping)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num_bins', type=int, default=15,
                        help='Number of temporal bins T (default: 15)')
    args = parser.parse_args()

    # ---- load trial-to-stimulus mapping from raw .mat ----
    print(f"Loading trial-to-stimulus mapping from: {args.raw_mat}")
    with h5py.File(args.raw_mat, 'r') as f:
        allmat = np.array(f['ALLMAT'])   # shape (6, n_trials)
    train_stim_ids = allmat[1].astype(int)   # values 1-22248 or 0
    test_stim_ids  = allmat[2].astype(int)   # values 1-100 or 0
    print(f"  ALLMAT shape: {allmat.shape}")
    print(f"  Unique train stim IDs (nonzero): {np.sum(train_stim_ids > 0)}")
    print(f"  Unique test  stim IDs (nonzero): {np.sum(test_stim_ids  > 0)}")

    unique_train = sorted(np.unique(train_stim_ids[train_stim_ids > 0]))
    unique_test  = sorted(np.unique(test_stim_ids[test_stim_ids > 0]))
    n_train = len(unique_train)
    n_test  = len(unique_test)
    print(f"  n_train stimuli: {n_train} | n_test stimuli: {n_test}")

    # ---- load normalized timeseries ----
    print(f"\nLoading timeseries HDF5: {args.timeseries_h5}")
    with h5py.File(args.timeseries_h5, 'r') as f:
        print(f"  Dataset shape: {f['timeseries_normalized'].shape}")
        normalized = f['timeseries_normalized'][:]   # (n_timepoints, n_electrodes, n_trials)
    print(f"  Loaded: {normalized.shape}  dtype: {normalized.dtype}")

    # ---- bin timepoints into T bins ----
    print(f"\nBinning {normalized.shape[0]} timepoints into {args.num_bins} bins...")
    binned = bin_timeseries(normalized, args.num_bins)   # (T, n_electrodes, n_trials)
    print(f"  Binned shape: {binned.shape}")
    del normalized  # free memory

    # ---- transpose to (n_trials, T, n_electrodes) ----
    binned = binned.transpose(2, 0, 1).astype(np.float32)
    print(f"  After transpose: {binned.shape}  (n_trials, T, n_electrodes)")

    n_electrodes = binned.shape[2]
    T = args.num_bins

    # ---- build train array: average over repetitions per stimulus ----
    print(f"\nAveraging train trials by stimulus ID...")
    train_ts = np.zeros((n_train, T, n_electrodes), dtype=np.float32)
    for stim_id in unique_train:
        mask = train_stim_ids == stim_id
        train_ts[stim_id - 1] = binned[mask].mean(axis=0)
    print(f"  train_timeseries: {train_ts.shape}")

    # ---- build test array: average over repetitions per stimulus ----
    print(f"Averaging test trials by stimulus ID...")
    test_ts = np.zeros((n_test, T, n_electrodes), dtype=np.float32)
    for stim_id in unique_test:
        mask = test_stim_ids == stim_id
        test_ts[stim_id - 1] = binned[mask].mean(axis=0)
    print(f"  test_timeseries:  {test_ts.shape}")
    del binned

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\nSaving to: {args.output_path}")
    with h5py.File(args.output_path, 'w') as f:
        # chunk along the trial axis for efficient single-trial reads at inference
        f.create_dataset(
            'train_timeseries', data=train_ts,
            chunks=(1, T, n_electrodes),
            compression='gzip', compression_opts=4,
        )
        f.create_dataset(
            'test_timeseries', data=test_ts,
            chunks=(1, T, n_electrodes),
            compression='gzip', compression_opts=4,
        )
        f.attrs['num_bins'] = args.num_bins
        f.attrs['n_train']  = n_train
        f.attrs['n_test']   = n_test

    print("\nDone! Verifying output...")
    with h5py.File(args.output_path, 'r') as f:
        print(f"  train_timeseries: {f['train_timeseries'].shape}")
        print(f"  test_timeseries:  {f['test_timeseries'].shape}")
        print(f"  num_bins: {f.attrs['num_bins']}")


if __name__ == '__main__':
    main()
