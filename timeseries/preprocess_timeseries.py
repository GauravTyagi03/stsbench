"""
Preprocess Timeseries Data for DDPM Conditioning — Ventral Stream (Both Monkeys)
==================================================================================
Fixes the electrode mismatch in the original single-monkey version:

  Original bug: only monkey F timeseries was processed, and electrodes were
  selected as indices 0..314, which are not the V4 channels used to build the
  pickle (monkey N V4: 512–768, monkey F V4: 832–1024, then reliability > 0.3).

  This script mirrors preprocess_ventral_dataset.py exactly:
    1. For each monkey, slices the correct V4 channel range from the timeseries HDF5
    2. Applies the same reliability threshold (> 0.3) using the paper_normalized.mat
    3. Averages over repetitions per stimulus ID using that monkey's ALLMAT
    4. Concatenates both monkeys along the electrode dimension

Row order matches the pickle (sorted stimulus ID), so image-neural pairings
in the dataloader remain correct.

Usage:
  python preprocess_timeseries.py \\
      --timeseries_h5_N /path/to/monkeyN_timeseries_normalized.h5 \\
      --timeseries_h5_F /path/to/monkeyF_timeseries_normalized.h5 \\
      --raw_mat_N       /path/to/monkeyN_THINGS_MUA_trials.mat \\
      --raw_mat_F       /path/to/monkeyF_THINGS_MUA_trials.mat \\
      --paper_norm_N    /path/to/monkeyN_paper_normalized.mat \\
      --paper_norm_F    /path/to/monkeyF_paper_normalized.mat \\
      --output_path     /path/to/ventral_stream_timeseries_preprocessed.h5 \\
      --num_bins        15
"""

import argparse
import os

import h5py
import numpy as np

# V4 channel ranges — must match preprocess_ventral_dataset.py
V4_RANGE_N = (512, 768)
V4_RANGE_F = (832, 1024)


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
    data = data[:usable]

    binned = data.reshape(num_bins, bin_width, data.shape[1], data.shape[2]).mean(axis=1)
    return binned  # (num_bins, n_electrodes, n_trials)


def process_monkey(
    timeseries_h5: str,
    raw_mat: str,
    paper_norm_mat: str,
    v4_range: tuple,
    reliability_threshold: float,
    num_bins: int,
    monkey_name: str,
):
    """
    Process one monkey's timeseries data, selecting V4 channels and applying
    reliability filtering to match the ventral stream pickle.

    Returns:
        train_ts: (n_train, T, n_reliable_v4)  float32
        test_ts:  (n_test,  T, n_reliable_v4)  float32
    """
    print(f"\n{'='*60}")
    print(f"Processing {monkey_name}  |  V4 channels {v4_range[0]}:{v4_range[1]}")
    print(f"{'='*60}")

    # ---- trial-to-stimulus mapping ----
    print(f"Loading trial mapping from: {raw_mat}")
    with h5py.File(raw_mat, 'r') as f:
        allmat = np.array(f['ALLMAT'])   # (6, n_trials)
    train_stim_ids = allmat[1].astype(int)   # 1-indexed or 0 = not a train trial
    test_stim_ids  = allmat[2].astype(int)

    unique_train = sorted(np.unique(train_stim_ids[train_stim_ids > 0]))
    unique_test  = sorted(np.unique(test_stim_ids[test_stim_ids > 0]))
    n_train = len(unique_train)
    n_test  = len(unique_test)
    print(f"  Unique train stimuli: {n_train} | Unique test stimuli: {n_test}")

    # ---- V4 channel selection ----
    print(f"Loading timeseries from: {timeseries_h5}")
    with h5py.File(timeseries_h5, 'r') as f:
        full_shape = f['timeseries_normalized'].shape
        print(f"  Full timeseries shape: {full_shape}  (n_timepoints, n_electrodes, n_trials)")
        # Load only V4 slice to save memory
        ts = f['timeseries_normalized'][:, v4_range[0]:v4_range[1], :]
    print(f"  After V4 slice: {ts.shape}  ({v4_range[1]-v4_range[0]} channels)")

    # ---- reliability filtering (mirrors preprocess_ventral_dataset.py) ----
    print(f"Loading reliability from: {paper_norm_mat}")
    with h5py.File(paper_norm_mat, 'r') as f:
        # reliab shape in h5py: (n_electrodes, n_repetitions) — average over reps
        reliab = np.mean(np.array(f['reliab'])[v4_range[0]:v4_range[1]], axis=1)
    reliable_mask = reliab > reliability_threshold
    n_reliable = int(reliable_mask.sum())
    print(f"  Electrodes passing reliability > {reliability_threshold}: {n_reliable} / {len(reliab)}")

    ts = ts[:, reliable_mask, :]   # (n_timepoints, n_reliable, n_trials)

    # ---- bin timepoints ----
    print(f"Binning {ts.shape[0]} timepoints into {num_bins} bins...")
    binned = bin_timeseries(ts, num_bins)   # (num_bins, n_reliable, n_trials)
    del ts
    print(f"  Binned shape: {binned.shape}")

    # ---- transpose to (n_trials, T, n_reliable) ----
    binned = binned.transpose(2, 0, 1).astype(np.float32)
    print(f"  After transpose: {binned.shape}  (n_trials, T, n_reliable)")

    T = num_bins

    # ---- average over repetitions per stimulus ----
    print("Averaging train trials by stimulus ID...")
    train_ts = np.zeros((n_train, T, n_reliable), dtype=np.float32)
    for stim_id in unique_train:
        mask = train_stim_ids == stim_id
        train_ts[stim_id - 1] = binned[mask].mean(axis=0)
    print(f"  train_timeseries: {train_ts.shape}")

    print("Averaging test trials by stimulus ID...")
    test_ts = np.zeros((n_test, T, n_reliable), dtype=np.float32)
    for stim_id in unique_test:
        mask = test_stim_ids == stim_id
        test_ts[stim_id - 1] = binned[mask].mean(axis=0)
    print(f"  test_timeseries:  {test_ts.shape}")

    del binned
    return train_ts, test_ts


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ventral stream timeseries for both monkeys'
    )
    parser.add_argument('--timeseries_h5_N', type=str, required=True,
                        help='Monkey N timeseries_normalized.h5')
    parser.add_argument('--timeseries_h5_F', type=str, required=True,
                        help='Monkey F timeseries_normalized.h5')
    parser.add_argument('--raw_mat_N', type=str, required=True,
                        help='Monkey N THINGS_MUA_trials.mat (trial-to-stimulus mapping)')
    parser.add_argument('--raw_mat_F', type=str, required=True,
                        help='Monkey F THINGS_MUA_trials.mat (trial-to-stimulus mapping)')
    parser.add_argument('--paper_norm_N', type=str, required=True,
                        help='Monkey N paper_normalized.mat (for reliability scores)')
    parser.add_argument('--paper_norm_F', type=str, required=True,
                        help='Monkey F paper_normalized.mat (for reliability scores)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num_bins', type=int, default=15,
                        help='Number of temporal bins T (default: 15)')
    parser.add_argument('--reliability_threshold', type=float, default=0.3,
                        help='Minimum reliability to include electrode (default: 0.3)')
    args = parser.parse_args()

    # ---- process each monkey ----
    train_N, test_N = process_monkey(
        timeseries_h5=args.timeseries_h5_N,
        raw_mat=args.raw_mat_N,
        paper_norm_mat=args.paper_norm_N,
        v4_range=V4_RANGE_N,
        reliability_threshold=args.reliability_threshold,
        num_bins=args.num_bins,
        monkey_name='monkeyN',
    )
    train_F, test_F = process_monkey(
        timeseries_h5=args.timeseries_h5_F,
        raw_mat=args.raw_mat_F,
        paper_norm_mat=args.paper_norm_F,
        v4_range=V4_RANGE_F,
        reliability_threshold=args.reliability_threshold,
        num_bins=args.num_bins,
        monkey_name='monkeyF',
    )

    # ---- concatenate along electrode dimension ----
    print(f"\n{'='*60}")
    print("Concatenating monkeys along electrode dimension...")
    train_ts = np.concatenate([train_N, train_F], axis=2)   # (n_train, T, N+F_reliable)
    test_ts  = np.concatenate([test_N,  test_F],  axis=2)   # (n_test,  T, N+F_reliable)
    print(f"  Combined train: {train_ts.shape}")
    print(f"  Combined test:  {test_ts.shape}")
    del train_N, train_F, test_N, test_F

    n_electrodes = train_ts.shape[2]
    T = args.num_bins

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\nSaving to: {args.output_path}")
    with h5py.File(args.output_path, 'w') as f:
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
        f.attrs['num_bins']             = args.num_bins
        f.attrs['reliability_threshold'] = args.reliability_threshold
        f.attrs['n_train']              = train_ts.shape[0]
        f.attrs['n_test']               = test_ts.shape[0]
        f.attrs['n_electrodes']         = n_electrodes

    print("\nDone! Verifying output...")
    with h5py.File(args.output_path, 'r') as f:
        print(f"  train_timeseries: {f['train_timeseries'].shape}")
        print(f"  test_timeseries:  {f['test_timeseries'].shape}")
        print(f"  num_bins:         {f.attrs['num_bins']}")
        print(f"  n_electrodes:     {f.attrs['n_electrodes']}")


if __name__ == '__main__':
    main()
