"""
Alternative Time-Series Normalization

This script implements a two-stage normalization approach:
1. Stage 1: Trial-wise baseline normalization using first 100ms
2. Stage 2: Per-(electrode, day, bin) normalization using ALL trials
   - Normalizes individual timepoints (NOT averaged bins)
   - Statistics computed from all timepoints within each (electrode, day, bin)
   - Uses all trials (test + train) to ensure dense, robust statistics

Outputs are written as HDF5 (.h5) for large arrays so you can load slices and reduce memory:
  - baseline: {monkey}_baseline_normalized.h5  dataset 'baseline_normalized', attr 'baseline_window'
  - final:    {monkey}_timeseries_normalized.h5 dataset 'timeseries_normalized', attrs 'bin_width', 'n_bins'
            Final output preserves timepoint granularity (shape: n_timepoints × n_electrodes × n_trials)
Use load_normalized_h5(path, key=..., load_slice=(slice(0,1000), slice(None), slice(None))) for partial reads.

Usage:
    python alternative_timeseries_norm.py --monkey monkeyF
    python alternative_timeseries_norm.py --monkey monkeyN --baseline_window 100 --bin_width 10
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
import os
import sys
import json
from typing import Tuple, Dict, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HDF5 chunk size along first axis for efficient partial reads (time/bins)
HDF5_CHUNK_AXIS0 = 2000


def _write_h5_array(f: "h5py.File", key: str, arr: np.ndarray, axis0_chunk: int = HDF5_CHUNK_AXIS0) -> None:
    """Create a chunked dataset and write array in blocks along axis 0 to limit memory during write."""
    n0 = arr.shape[0]
    chunk_len = min(axis0_chunk, n0) if n0 else 1

    # HDF5 has a hard limit of 4GB per chunk (in bytes)
    max_chunk_bytes = 4 * 1024**3  # 4GB
    max_chunk_elements = max_chunk_bytes // arr.dtype.itemsize

    # Start with initial chunks
    chunks = [chunk_len] + list(arr.shape[1:])

    # Calculate total elements in proposed chunk
    total_elements = np.prod(chunks)

    # If chunk is too large, reduce dimensions to fit under limit
    while total_elements > max_chunk_elements:
        # Reduce the last dimension first (trials), then work backwards
        for i in range(len(chunks) - 1, 0, -1):  # Don't modify first dimension
            if chunks[i] > 1:
                # Reduce by half or to 1, whichever is larger
                chunks[i] = max(1, chunks[i] // 2)
                total_elements = np.prod(chunks)
                if total_elements <= max_chunk_elements:
                    break
        # Safety check: if we still can't fit, reduce first dimension too
        if total_elements > max_chunk_elements and chunks[0] > 1:
            chunks[0] = max(1, chunks[0] // 2)
            total_elements = np.prod(chunks)

    chunks = tuple(chunks)
    print(f"  Creating dataset '{key}' with shape {arr.shape}, chunks {chunks}, "
          f"chunk size: {total_elements * arr.dtype.itemsize / 1024**2:.1f} MB")

    dset = f.create_dataset(key, shape=arr.shape, dtype=arr.dtype, chunks=chunks, compression='gzip')
    step = chunk_len
    for start in range(0, n0, step):
        end = min(start + step, n0)
        dset[start:end] = arr[start:end]


def load_normalized_h5(
    filepath: str,
    key: str = 'timeseries_normalized',
    load_slice: Optional[Tuple[Union[slice, int], ...]] = None,
) -> np.ndarray:
    """Load normalized data from an HDF5 file (optionally a slice to reduce memory).

    Args:
        filepath: Path to .h5 file.
        key: Dataset name ('timeseries_normalized' or 'baseline_normalized').
        load_slice: Optional tuple of slices, e.g. (slice(0, 1000), slice(None), slice(0, 10))
                    to load only part of the array. If None, loads the full array.

    Returns:
        NumPy array (full or sliced). Attributes (bin_width, n_bins, baseline_window) are
        stored on the file's root or the dataset; access via h5py.File(filepath)['/'].attrs.
    """
    with h5py.File(filepath, 'r') as f:
        dset = f[key]
        if load_slice is not None:
            return dset[load_slice]
        return dset[:]


class TimeseriesNormalization:
    """
    Two-stage baseline + temporal bin normalization.

    Stage 1: Normalize each trial using its own baseline period
    Stage 2: Normalize individual timepoints per (electrode, day, bin) using ALL trials
            - Statistics computed from all timepoints within each (electrode, day, bin)
            - Uses all trials (test + train) to ensure dense, robust statistics
            - No averaging of timepoints - output preserves temporal granularity
    """

    def __init__(self, baseline_window: int = 100, bin_width: int = 10, tb: Optional[np.ndarray] = None):
        """
        Args:
            baseline_window: Number of timepoints for baseline (default: 100ms)
            bin_width: Width of temporal bins in timepoints (default: 10ms)
            tb: Time base array in milliseconds (optional)
        """
        self.baseline_window = baseline_window
        self.bin_width = bin_width
        self.tb = tb

    def stage1_baseline_norm(self, allmua: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Stage 1: Trial-wise baseline normalization (in-place).

        Normalize each trial using its own baseline period (first N timepoints).
        Modifies allmua in place to avoid a second full-size copy.

        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)

        Returns:
            allmua: same array, now baseline-normalized (n_timepoints, n_electrodes, n_trials)
            stats: Dictionary with baseline statistics
        """
        print("\nStage 1: Baseline normalization (in-place)")
        print(f"  Using first {self.baseline_window} timepoints as baseline")

        _, n_electrodes, n_trials = allmua.shape

        # Extract baseline period using pre-stimulus timepoints if tb is available
        if self.tb is not None:
            # Find all timepoints where t <= 0 (pre-stimulus)
            # Use contiguous slicing to avoid memory copy when possible
            baseline_end = np.searchsorted(self.tb, 0, side='right')
            if baseline_end == 0:
                # No pre-stimulus timepoints found
                baseline_end = min(self.baseline_window, allmua.shape[0])
                print(f"  Warning: No pre-stimulus timepoints found, using first {baseline_end} timepoints")
            # Use slicing (creates view, not copy) since baseline is at start
            baseline_data = allmua[:baseline_end, :, :]
            print(f"  Using {baseline_end} pre-stimulus timepoints (t <= 0) as baseline")
            print(f"  Time range: {self.tb[0]:.1f} to {self.tb[baseline_end-1]:.1f} ms")
        else:
            # Fallback to old behavior
            baseline_data = allmua[:self.baseline_window, :, :]
            print(f"  Warning: No time base provided, using first {self.baseline_window} timepoints")

        # Compute mean and std for each (electrode, trial)
        baseline_mean = baseline_data.mean(axis=0, keepdims=True)  # (1, n_electrodes, n_trials)
        baseline_std = baseline_data.std(axis=0, keepdims=True)    # (1, n_electrodes, n_trials)

        # Handle zero std
        zero_std_count = np.sum(baseline_std == 0)
        baseline_std[baseline_std == 0] = 1.0

        # Normalize in place to avoid allocating a second full-size array
        allmua -= baseline_mean
        allmua /= baseline_std

        stats = {
            'baseline_window': self.baseline_window,
            'zero_std_count': int(zero_std_count),
            'zero_std_percentage': 100 * zero_std_count / (n_electrodes * n_trials),
            'mean_baseline_mean': float(baseline_mean.mean()),
            'mean_baseline_std': float(baseline_std.mean())
        }

        print(f"  Zero std cases: {zero_std_count} ({stats['zero_std_percentage']:.2f}%)")
        print(f"  Mean baseline mean: {stats['mean_baseline_mean']:.4f}")
        print(f"  Mean baseline std: {stats['mean_baseline_std']:.4f}")

        # Verify baseline normalization
        baseline_check = allmua[:self.baseline_window, :, :]
        print(f"  Verification - baseline period mean: {baseline_check.mean():.6f}")
        print(f"  Verification - baseline period std: {baseline_check.std():.6f}")

        return allmua, stats


    def stage2_bin_norm(
        self,
        baseline_normalized: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Stage 2: Per-(electrode, day, bin) normalization using ALL trials.

        Simpler approach: Use all available trials for statistics instead of
        test-only pool (which is too sparse).

        Args:
            baseline_normalized: (n_timepoints, n_electrodes, n_trials)
            allmat: (6, n_trials)

        Returns:
            normalized: (truncated_length, n_electrodes, n_trials) where
                       truncated_length = n_bins * bin_width
            stats: Dictionary with normalization statistics
        """
        print("\nStage 2: Per-(electrode, day, bin) normalization (using ALL trials)")

        n_timepoints, n_electrodes, n_trials = baseline_normalized.shape

        # Truncate to fit exact number of bins
        n_bins = n_timepoints // self.bin_width
        truncated_length = n_bins * self.bin_width
        data = baseline_normalized[:truncated_length, :, :]

        print(f"  Original timepoints: {n_timepoints}")
        print(f"  Bin width: {self.bin_width}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Truncated length: {truncated_length}")
        print(f"  Data shape after truncation: {data.shape}")

        # Create bin assignment for each timepoint
        bin_ids = np.arange(truncated_length) // self.bin_width
        print(f"  Bin IDs shape: {bin_ids.shape}")
        print(f"  Bin IDs range: {bin_ids.min()} to {bin_ids.max()}")

        # Extract metadata
        days = allmat[5].astype(np.int32)
        unique_days = np.unique(days)

        print(f"  Electrodes: {n_electrodes}")
        print(f"  Days: {len(unique_days)} unique days: {unique_days}")
        print(f"  Total trials: {n_trials}")
        print(f"  Days array shape: {days.shape}")

        # Initialize normalized array
        normalized = np.zeros_like(data)
        print(f"  Initialized normalized array shape: {normalized.shape}")

        # Statistics tracking
        stats = {
            'n_electrodes': n_electrodes,
            'n_days': len(unique_days),
            'n_bins': n_bins,
            'zero_std': 0,
            'total_combinations': n_electrodes * len(unique_days) * n_bins,
            'sample_means': [],
            'sample_stds': [],
            'sample_input_ranges': [],
            'sample_output_ranges': []
        }

        print("\n  Starting normalization loop...")
        # Normalize each (electrode, day, bin) using ALL trials from that day
        for elec_idx in range(n_electrodes):
            if elec_idx % 100 == 0:
                print(f"\n  --- Processing electrode {elec_idx}/{n_electrodes} ---")

            for day_idx, day in enumerate(unique_days):
                day_mask = (days == day)
                n_trials_for_day = day_mask.sum()

                # Verbose logging for first electrode and first day
                if elec_idx % 100 == 0 and day_idx == 0:
                    print(f"    Day {day}: {n_trials_for_day} trials (mask shape: {day_mask.shape}, sum: {n_trials_for_day})")

                for bin_idx in range(n_bins):
                    # Get timepoint mask for this bin
                    timepoint_mask = (bin_ids == bin_idx)
                    n_timepoints_in_bin = timepoint_mask.sum()

                    # Verbose logging for first electrode, first day, first 3 bins
                    verbose = (elec_idx % 100 == 0 and day_idx == 0 and bin_idx < 3)

                    if verbose:
                        print(f"      Bin {bin_idx}:")
                        print(f"        Timepoint mask sum: {n_timepoints_in_bin} (expected: {self.bin_width})")
                        print(f"        Timepoint indices in bin: {np.where(timepoint_mask)[0][:5]}...")

                    # Extract ALL timepoints for this (electrode, day, bin)
                    # Method 1: Using fancy indexing
                    # First: data[timepoint_mask, elec_idx, :] -> (n_timepoints_in_bin, n_trials)
                    temp_data = data[timepoint_mask, elec_idx, :]
                    if verbose:
                        print(f"        After timepoint mask: shape = {temp_data.shape}")
                        print(f"        Data range before day mask: [{temp_data.min():.4f}, {temp_data.max():.4f}]")
                        print(f"        Non-zero fraction: {(np.abs(temp_data) > 1e-6).sum() / temp_data.size:.2%}")

                    # Second: [:, day_mask] -> (n_timepoints_in_bin, n_trials_for_day)
                    day_bin_timepoints = temp_data[:, day_mask]

                    if verbose:
                        print(f"        After day mask: shape = {day_bin_timepoints.shape}")
                        print(f"        Expected shape: ({n_timepoints_in_bin}, {n_trials_for_day})")
                        print(f"        Data range: [{day_bin_timepoints.min():.4f}, {day_bin_timepoints.max():.4f}]")
                        print(f"        Sample values: {day_bin_timepoints.flatten()[:10]}")

                    # Compute mean/std from ALL timepoints and trials
                    mean = day_bin_timepoints.mean()
                    std = day_bin_timepoints.std()

                    if verbose:
                        print(f"        Computed mean: {mean:.6f}, std: {std:.6f}")

                    # Store sample statistics
                    if elec_idx % 100 == 0 and bin_idx % 10 == 0:
                        stats['sample_means'].append(float(mean))
                        stats['sample_stds'].append(float(std))
                        stats['sample_input_ranges'].append([float(day_bin_timepoints.min()),
                                                             float(day_bin_timepoints.max())])

                    if std == 0 or np.isnan(std):
                        if verbose:
                            print(f"        WARNING: Zero or NaN std, setting to 1.0")
                        std = 1.0
                        stats['zero_std'] += 1

                    # Normalize
                    normalized_values = (day_bin_timepoints - mean) / std

                    if verbose:
                        print(f"        Normalized range: [{normalized_values.min():.4f}, {normalized_values.max():.4f}]")
                        print(f"        Normalized mean: {normalized_values.mean():.6f}, std: {normalized_values.std():.6f}")
                        print(f"        Sample normalized values: {normalized_values.flatten()[:10]}")

                    # Assign back to normalized array
                    normalized[timepoint_mask, elec_idx, :][:, day_mask] = normalized_values

                    # Verify assignment worked
                    if verbose:
                        check_values = normalized[timepoint_mask, elec_idx, :][:, day_mask]
                        print(f"        Verification - retrieved from normalized array:")
                        print(f"          Shape: {check_values.shape}")
                        print(f"          Range: [{check_values.min():.4f}, {check_values.max():.4f}]")
                        print(f"          Mean: {check_values.mean():.6f}, Std: {check_values.std():.6f}")
                        print(f"          Non-zero fraction: {(np.abs(check_values) > 1e-6).sum() / check_values.size:.2%}")

                        if bin_idx % 10 == 0:
                            stats['sample_output_ranges'].append([float(check_values.min()),
                                                                  float(check_values.max())])

        # Calculate percentages
        stats['zero_std_percentage'] = 100 * stats['zero_std'] / stats['total_combinations']

        print(f"\n  Stage 2 Statistics:")
        print(f"    Zero std cases: {stats['zero_std']} ({stats['zero_std_percentage']:.2f}%)")
        if stats['sample_means']:
            print(f"    Sample means (first 10): {stats['sample_means'][:10]}")
            print(f"    Sample stds (first 10): {stats['sample_stds'][:10]}")
            print(f"    Sample input ranges (first 5): {stats['sample_input_ranges'][:5]}")
            print(f"    Sample output ranges (first 5): {stats['sample_output_ranges'][:5]}")

        # Verify normalization
        print(f"\n  Normalized data verification:")
        print(f"    Mean: {normalized.mean():.6f}")
        print(f"    Std: {normalized.std():.6f}")
        print(f"    Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
        print(f"    Non-zero fraction: {(np.abs(normalized) > 1e-6).sum() / normalized.size:.2%}")

        # Check a random slice
        random_slice = normalized[:100, 0, :10]
        print(f"\n  Random slice check (first 100 timepoints, electrode 0, first 10 trials):")
        print(f"    Shape: {random_slice.shape}")
        print(f"    Mean: {random_slice.mean():.6f}, Std: {random_slice.std():.6f}")
        print(f"    Range: [{random_slice.min():.4f}, {random_slice.max():.4f}]")
        print(f"    Non-zero fraction: {(np.abs(random_slice) > 1e-6).sum() / random_slice.size:.2%}")

        return normalized, stats

    def fit_transform(
        self,
        allmua: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete two-stage normalization pipeline.

        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)
            allmat: (6, n_trials)

        Returns:
            baseline_normalized: (n_timepoints, n_electrodes, n_trials)
            final_normalized: (truncated_length, n_electrodes, n_trials) where
                            truncated_length = n_bins * bin_width
            metadata: Combined statistics from both stages
        """
        # Stage 1: Baseline normalization
        baseline_normalized, stage1_stats = self.stage1_baseline_norm(allmua)

        # Stage 2: Bin normalization (operates on individual timepoints, not averages)
        final_normalized, stage2_stats = self.stage2_bin_norm(baseline_normalized, allmat)

        # Combine metadata
        metadata = {
            'baseline_window': self.baseline_window,
            'bin_width': self.bin_width,
            'stage1': stage1_stats,
            'stage2': stage2_stats,
            'final_shape': {
                'baseline_normalized': list(baseline_normalized.shape),
                'final_normalized': list(final_normalized.shape)
            }
        }

        return baseline_normalized, final_normalized, metadata


def load_mua_data(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MUA data and apply channel mapping.

    Args:
        data_dir: Path to data directory
        monkey_name: 'monkeyF' or 'monkeyN'

    Returns:
        allmua: (n_timepoints, n_electrodes, n_trials)
        allmat: (6, n_trials)
        tb: (n_timepoints,) time base in milliseconds
    """
    print(f"Loading data for {monkey_name}...")

    # Load main data file
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')

    with h5py.File(data_file, 'r') as f:
        # Load ALLMUA, ALLMAT, and tb (single read)
        data = np.array(f['ALLMUA'])
        allmat = np.array(f['ALLMAT'])
        tb = np.array(f['tb']).flatten()

    print(f"Loaded ALLMUA shape: {data.shape}")
    print(f"Loaded ALLMAT shape: {allmat.shape}")

    # Load channel mapping - flexible search for any mapping file
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    if not mapping_files:
        raise FileNotFoundError(f"Could not find mapping file for {monkey_name} in {data_dir}")

    mapping_file = os.path.join(data_dir, mapping_files[0])
    print(f"Using mapping file: {mapping_files[0]}")

    mapping_data = loadmat(mapping_file)
    mapping = mapping_data['mapping'].flatten() - 1  # Convert to 0-indexed

    # Apply mapping and transpose in one step to keep only one full-size copy in memory
    # h5py loads as (n_timepoints, n_trials, n_electrodes) -> we want (n_timepoints, n_electrodes, n_trials)
    print(f"Applying channel mapping and transpose (single copy)...")
    allmua = np.ascontiguousarray(data[..., mapping].transpose(0, 2, 1))
    del data

    # Use float32 to halve memory (sufficient precision for normalization)
    allmua = allmua.astype(np.float32)
    print(f"Final ALLMUA shape: {allmua.shape}, dtype: {allmua.dtype}")
    print(f"Time base: {tb.min():.1f} to {tb.max():.1f} ms ({len(tb)} timepoints)")
    return allmua, allmat, tb


def main():
    parser = argparse.ArgumentParser(
        description='Alternative time-series normalization with baseline and binning'
    )
    parser.add_argument('--monkey', type=str, required=True,
                        choices=['monkeyF', 'monkeyN'],
                        help='Monkey name')
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/groups/anishm/tvsd/',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/',
                        help='Path to output directory')
    parser.add_argument('--baseline_window', type=int, default=100,
                        help='Number of baseline timepoints (default: 100)')
    parser.add_argument('--bin_width', type=int, default=10,
                        help='Bin width in timepoints (default: 10)')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("ALTERNATIVE TIME-SERIES NORMALIZATION")
    print("="*60)
    print(f"Monkey: {args.monkey}")
    print(f"Baseline window: {args.baseline_window} timepoints")
    print(f"Bin width: {args.bin_width} timepoints")
    print("="*60)

    # Load data
    allmua, allmat, tb = load_mua_data(args.data_dir, args.monkey)

    # Initialize normalizer (tb is stored in normalizer, so we can delete the reference)
    normalizer = TimeseriesNormalization(
        baseline_window=args.baseline_window,
        bin_width=args.bin_width,
        tb=tb
    )

    # Keep a reference to tb for saving later (normalizer.tb is a reference, not a copy)
    tb_for_saving = normalizer.tb

    # Normalize
    print("\nApplying two-stage normalization...")
    baseline_normalized, final_normalized, metadata = normalizer.fit_transform(allmua, allmat)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline-normalized shape: {baseline_normalized.shape}")
    print(f"Final normalized shape: {final_normalized.shape}")

    # Check for invalid values
    if np.any(np.isnan(final_normalized)):
        print("WARNING: NaN values detected in final normalized data!")
        metadata['has_nan'] = True
    else:
        metadata['has_nan'] = False

    if np.any(np.isinf(final_normalized)):
        print("WARNING: Inf values detected in final normalized data!")
        metadata['has_inf'] = True
    else:
        metadata['has_inf'] = False

    max_abs_value = np.abs(final_normalized).max()
    print(f"Max absolute value: {max_abs_value:.4f}")
    if max_abs_value > 10:
        print("WARNING: Very large values detected (> 10)")

    metadata['max_abs_value'] = float(max_abs_value)
    metadata['final_mean'] = float(final_normalized.mean())
    metadata['final_std'] = float(final_normalized.std())

    print(f"Final mean: {metadata['final_mean']:.6f}")
    print(f"Final std: {metadata['final_std']:.6f}")

    # Save baseline-normalized data (intermediate result) as HDF5 for large arrays and partial loading
    baseline_file = os.path.join(args.output_dir, f'{args.monkey}_baseline_normalized.h5')
    print(f"\nSaving baseline-normalized data to {baseline_file}")
    with h5py.File(baseline_file, 'w') as f:
        _write_h5_array(f, 'baseline_normalized', baseline_normalized)
        f.create_dataset('tb', data=tb_for_saving)
        f.attrs['baseline_window'] = args.baseline_window
    del baseline_normalized  # Free full-size array before saving final outputs

    # Save final normalized data as HDF5
    final_file = os.path.join(args.output_dir, f'{args.monkey}_timeseries_normalized.h5')
    print(f"Saving final normalized data to {final_file}")
    with h5py.File(final_file, 'w') as f:
        _write_h5_array(f, 'timeseries_normalized', final_normalized)
        # Save truncated time base matching final_normalized length
        f.create_dataset('tb', data=tb_for_saving[:final_normalized.shape[0]])
        f.attrs['bin_width'] = args.bin_width
        n_bins = final_normalized.shape[0] // args.bin_width
        f.attrs['n_bins'] = n_bins
        f.attrs['n_timepoints'] = final_normalized.shape[0]
        f.attrs['note'] = 'Output preserves timepoint granularity (not averaged into bins)'

    # Save metadata as JSON
    metadata_file = os.path.join(args.output_dir, f'{args.monkey}_normalization_metadata.json')
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
