"""
Preprocess Timeseries Data for DDPM Conditioning
=================================================
Reads the normalized timeseries HDF5 (from normalization/alternative_timeseries_norm.py),
bins individual timepoints into T equal-width bins, and splits into train/test
according to the original dataset pickle.

Input:
  {monkey}_timeseries_normalized.h5   (from normalization/)
      dataset 'timeseries_normalized'  shape: (n_timepoints, n_electrodes, n_trials)
  {name}_dataset.pickle               (from dataset/)
      used to determine n_train and n_test

Output:
  {name}_timeseries_preprocessed.h5   with:
      'train_timeseries'  shape: (n_train, T, n_electrodes)  float32
      'test_timeseries'   shape: (n_test,  T, n_electrodes)  float32
  attrs: num_bins, n_train, n_test

IMPORTANT — trial ordering assumption:
  The normalized HDF5 must contain trials in the same order as the pickle:
  the first n_train trials are training trials, the remaining n_test are test.
  Verify this matches your data before training. You can override n_train
  with --n_train if the split differs from what the pickle reports.

Usage:
  python preprocess_timeseries.py \\
      --timeseries_h5  /path/to/monkeyF_timeseries_normalized.h5 \\
      --pickle_path    /path/to/dorsal_stream_dataset.pickle \\
      --output_path    /path/to/dorsal_stream_timeseries_preprocessed.h5 \\
      --num_bins       15
"""

import argparse
import os
import pickle

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
    parser.add_argument('--pickle_path', type=str, required=True,
                        help='Path to {name}_dataset.pickle (to infer train/test split sizes)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num_bins', type=int, default=15,
                        help='Number of temporal bins T (default: 15)')
    parser.add_argument('--n_train', type=int, default=None,
                        help='Override number of train trials (default: inferred from pickle)')
    args = parser.parse_args()

    # ---- determine train/test split sizes from pickle ----
    print(f"Loading pickle: {args.pickle_path}")
    with open(args.pickle_path, 'rb') as f:
        data = pickle.load(f)
    n_train = args.n_train if args.n_train is not None else data['train_activity'].shape[0]
    n_test = data['test_activity'].shape[0]
    print(f"  Train trials: {n_train} | Test trials: {n_test} | Total: {n_train + n_test}")

    # ---- load normalized timeseries ----
    print(f"\nLoading timeseries HDF5: {args.timeseries_h5}")
    with h5py.File(args.timeseries_h5, 'r') as f:
        print(f"  Dataset shape: {f['timeseries_normalized'].shape}")
        normalized = f['timeseries_normalized'][:]   # (n_timepoints, n_electrodes, n_trials)
    print(f"  Loaded: {normalized.shape}  dtype: {normalized.dtype}")

    n_total_trials = normalized.shape[2]
    if n_total_trials != n_train + n_test:
        raise ValueError(
            f"Trial count mismatch: HDF5 has {n_total_trials} trials, "
            f"pickle expects {n_train + n_test} (train={n_train}, test={n_test}).\n"
            f"Either the files are mismatched, or use --n_train to override the split."
        )

    # ---- bin timepoints into T bins ----
    print(f"\nBinning {normalized.shape[0]} timepoints into {args.num_bins} bins...")
    binned = bin_timeseries(normalized, args.num_bins)   # (T, n_electrodes, n_trials)
    print(f"  Binned shape: {binned.shape}")
    del normalized  # free memory

    # ---- transpose to (n_trials, T, n_electrodes) ----
    binned = binned.transpose(2, 0, 1).astype(np.float32)
    print(f"  After transpose: {binned.shape}  (n_trials, T, n_electrodes)")

    # ---- split into train / test ----
    train_ts = binned[:n_train]
    test_ts  = binned[n_train:n_train + n_test]
    print(f"\n  train_timeseries: {train_ts.shape}")
    print(f"  test_timeseries:  {test_ts.shape}")
    del binned

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\nSaving to: {args.output_path}")
    with h5py.File(args.output_path, 'w') as f:
        # chunk along the trial axis for efficient single-trial reads at inference
        f.create_dataset(
            'train_timeseries', data=train_ts,
            chunks=(1, args.num_bins, train_ts.shape[2]),
            compression='gzip', compression_opts=4,
        )
        f.create_dataset(
            'test_timeseries', data=test_ts,
            chunks=(1, args.num_bins, test_ts.shape[2]),
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
