"""
Verify Paper's Normalization Method

This script implements the exact normalization method from the paper and validates
it against the provided THINGS_normMUA.mat file.

Key insight: The paper normalizes against TEST pool statistics, not training statistics.

Usage:
    python verify_paper_normalization.py --monkey monkeyF
    python verify_paper_normalization.py --monkey monkeyN --data_dir /path/to/data
"""

import argparse
import h5py
import numpy as np
import scipy.io
from scipy.io import loadmat, savemat
from scipy.stats import ks_2samp
import os
import sys
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import loadmat as load_matlab_file


def load_mat_file(filepath: str) -> Dict:
    """
    Load MATLAB file, handling both v7.2 and v7.3 (HDF5) formats.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        Dictionary with MATLAB variables
    """
    try:
        # Try loading as v7.2 format first
        return loadmat(filepath)
    except NotImplementedError as e:
        if 'v7.3' in str(e) or 'HDF' in str(e):
            # File is v7.3 format, use h5py
            print(f"Loading {filepath} as MATLAB v7.3 (HDF5) format...")
            data = {}
            with h5py.File(filepath, 'r') as f:
                # MATLAB v7.3 files store variables in the root group
                # Iterate through all keys and load datasets
                for key in f.keys():
                    # Skip MATLAB internal metadata (keys starting with #)
                    if not key.startswith('#'):
                        obj = f[key]
                        if isinstance(obj, h5py.Dataset):
                            # Direct dataset - load as numpy array
                            # Note: MATLAB stores matrices in column-major order,
                            # but h5py loads them correctly, so we may need to transpose
                            arr = np.array(obj)
                            # MATLAB v7.3 stores data in row-major when read via h5py
                            # but the shape might need adjustment depending on how it was saved
                            data[key] = arr
                        # Groups (structures) are skipped for now
            return data
        else:
            raise


class PaperNormalization:
    """
    Implements the paper's exact normalization method.

    Normalization steps:
    1. Average response across images in region-specific time windows
    2. For each (day, electrode), compute mean and std from TEST images only
    3. Normalize ALL trials using test-pool statistics
    """

    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        self.brain_regions = self._define_brain_regions()

    def _define_brain_regions(self) -> Dict[str, Dict]:
        """Define brain regions and time windows for each monkey."""
        if self.monkey_name == 'monkeyF':
            return {
                'V1': {'electrodes': range(0, 512), 'time_window': (25, 125)},
                'IT': {'electrodes': range(512, 832), 'time_window': (75, 175)},
                'V4': {'electrodes': range(832, 1024), 'time_window': (50, 150)}
            }
        elif self.monkey_name == 'monkeyN':
            return {
                'V1': {'electrodes': range(0, 512), 'time_window': (25, 125)},
                'V4': {'electrodes': range(512, 768), 'time_window': (50, 150)},
                'IT': {'electrodes': range(768, 1024), 'time_window': (75, 175)}
            }
        else:
            raise ValueError(f"Unknown monkey: {self.monkey_name}")

    def _get_brain_region(self, electrode_idx: int) -> str:
        """Get brain region for a given electrode index."""
        for region, info in self.brain_regions.items():
            if electrode_idx in info['electrodes']:
                return region
        raise ValueError(f"Electrode {electrode_idx} not found in any brain region")

    def extract_time_window(self, allmua: np.ndarray, electrode_idx: int) -> np.ndarray:
        """
        Extract and average MUA data in region-specific time window.

        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)
            electrode_idx: Index of electrode

        Returns:
            (n_trials,) averaged response in time window
        """
        region = self._get_brain_region(electrode_idx)
        start, end = self.brain_regions[region]['time_window']

        # Extract time window and average
        window_data = allmua[start:end, electrode_idx, :]
        return window_data.mean(axis=0)

    def normalize_electrode_day(
        self,
        allmua: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize each electrode separately for each day using test pool statistics.

        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)
            allmat: (6, n_trials) metadata array

        Returns:
            normalized: (n_electrodes, n_trials)
            stats: Dictionary with normalization statistics
        """
        n_electrodes = allmua.shape[1]
        n_trials = allmua.shape[2]

        # Extract metadata
        train_idx = allmat[1].astype(np.int32)
        test_idx = allmat[2].astype(np.int32)
        days = allmat[5].astype(np.int32)

        # Create masks
        train_mask = train_idx > 0
        test_mask = test_idx > 0

        unique_days = np.unique(days)

        # Initialize normalized array
        normalized = np.zeros((n_electrodes, n_trials))

        # Statistics tracking
        stats = {
            'n_electrodes': n_electrodes,
            'n_trials': n_trials,
            'n_days': len(unique_days),
            'no_test_trials': 0,
            'zero_std': 0
        }

        print(f"Normalizing {n_electrodes} electrodes across {len(unique_days)} days...")

        # Normalize each electrode
        for elec_idx in range(n_electrodes):
            if elec_idx % 100 == 0:
                print(f"  Processing electrode {elec_idx}/{n_electrodes}")

            # Extract time-windowed data for this electrode
            elec_data = self.extract_time_window(allmua, elec_idx)

            # Normalize separately for each day
            for day in unique_days:
                day_mask = (days == day)
                test_day_mask = day_mask & test_mask

                # Compute stats from TEST trials only
                test_data = elec_data[test_day_mask]

                if len(test_data) == 0:
                    # No test trials for this (electrode, day)
                    normalized[elec_idx, day_mask] = 0
                    stats['no_test_trials'] += 1
                    continue

                mean = test_data.mean()
                std = test_data.std()

                if std == 0 or np.isnan(std):
                    std = 1.0
                    stats['zero_std'] += 1

                # Normalize ALL trials in this day using test pool statistics
                normalized[elec_idx, day_mask] = (elec_data[day_mask] - mean) / std

        print(f"Normalization complete. Stats: {stats}")
        return normalized, stats

    def average_by_stimulus(
        self,
        normalized: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average normalized responses by stimulus ID.

        Args:
            normalized: (n_electrodes, n_trials)
            allmat: (6, n_trials)

        Returns:
            train_MUA: (n_train_stimuli, n_electrodes)
            test_MUA: (n_test_stimuli, n_electrodes)
            test_MUA_reps: (n_electrodes, n_test_stimuli, max_reps)
        """
        train_idx = allmat[1].astype(np.int32)
        test_idx = allmat[2].astype(np.int32)

        # Get unique stimulus IDs
        unique_train = np.unique(train_idx[train_idx > 0])
        unique_test = np.unique(test_idx[test_idx > 0])

        n_electrodes = normalized.shape[0]
        n_train_stim = len(unique_train)
        n_test_stim = len(unique_test)

        print(f"Averaging: {n_train_stim} train stimuli, {n_test_stim} test stimuli")

        # Initialize arrays
        train_MUA = np.zeros((n_train_stim, n_electrodes))
        test_MUA = np.zeros((n_test_stim, n_electrodes))

        # Average train stimuli
        for i, stim_id in enumerate(unique_train):
            stim_trials = (train_idx == stim_id)
            train_MUA[i, :] = normalized[:, stim_trials].mean(axis=1)

        # Average test stimuli
        for i, stim_id in enumerate(unique_test):
            stim_trials = (test_idx == stim_id)
            test_MUA[i, :] = normalized[:, stim_trials].mean(axis=1)

        # Create test_MUA_reps (with repetitions preserved)
        # Find max number of repetitions
        max_reps = max([np.sum(test_idx == stim_id) for stim_id in unique_test])
        test_MUA_reps = np.zeros((n_electrodes, n_test_stim, max_reps))

        for i, stim_id in enumerate(unique_test):
            stim_trials = np.where(test_idx == stim_id)[0]
            n_reps = len(stim_trials)
            test_MUA_reps[:, i, :n_reps] = normalized[:, stim_trials]

        return train_MUA, test_MUA, test_MUA_reps

    def fit_transform(
        self,
        allmua: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Complete normalization pipeline.

        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)
            allmat: (6, n_trials)

        Returns:
            train_MUA: (n_train_stimuli, n_electrodes)
            test_MUA: (n_test_stimuli, n_electrodes)
            test_MUA_reps: (n_electrodes, n_test_stimuli, n_reps)
            stats: Normalization statistics
        """
        # Step 1: Normalize per (electrode, day) using test pool
        normalized, stats = self.normalize_electrode_day(allmua, allmat)

        # Step 2: Average by stimulus
        train_MUA, test_MUA, test_MUA_reps = self.average_by_stimulus(normalized, allmat)

        return train_MUA, test_MUA, test_MUA_reps, stats


def load_mua_data(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MUA data and apply channel mapping.

    Args:
        data_dir: Path to data directory
        monkey_name: 'monkeyF' or 'monkeyN'

    Returns:
        allmua: (n_timepoints, n_electrodes, n_trials)
        allmat: (6, n_trials)
    """
    print(f"Loading data for {monkey_name}...")

    # Load main data file
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')

    with h5py.File(data_file, 'r') as f:
        # Load ALLMUA and ALLMAT
        allmua = np.array(f['ALLMUA'])
        allmat = np.array(f['ALLMAT'])

    print(f"Loaded ALLMUA shape: {allmua.shape}")
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

    print(f"Applying channel mapping...")
    allmua = allmua[..., mapping]

    # h5py loads data as (n_timepoints, n_trials, n_electrodes)
    # Need to transpose to (n_timepoints, n_electrodes, n_trials)
    print(f"Transposing from {allmua.shape} to (timepoints, electrodes, trials)")
    allmua = np.transpose(allmua, (0, 2, 1))

    print(f"Final ALLMUA shape: {allmua.shape}")
    return allmua, allmat


def validate_reconstruction(
    our_train: np.ndarray,
    our_test: np.ndarray,
    original_train: np.ndarray,
    original_test: np.ndarray,
    monkey_name: str,
    output_dir: str
) -> Dict:
    """
    Comprehensive validation of our reconstruction against original.

    Args:
        our_train: Our normalized training data
        our_test: Our normalized test data
        original_train: Original training data from THINGS_normMUA.mat
        original_test: Original test data from THINGS_normMUA.mat
        monkey_name: Monkey name for output files
        output_dir: Directory for saving results

    Returns:
        validation_stats: Dictionary with validation metrics
    """
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    validation_stats = {}

    # Global correlation
    train_corr = np.corrcoef(our_train.flatten(), original_train.flatten())[0, 1]
    test_corr = np.corrcoef(our_test.flatten(), original_test.flatten())[0, 1]

    print(f"\nGlobal Correlation:")
    print(f"  Train: {train_corr:.6f}")
    print(f"  Test:  {test_corr:.6f}")

    validation_stats['train_global_corr'] = train_corr
    validation_stats['test_global_corr'] = test_corr

    # Per-electrode correlation
    n_electrodes = our_train.shape[1]
    train_elec_corrs = []
    test_elec_corrs = []

    for elec in range(n_electrodes):
        if original_train[:, elec].std() > 0:
            corr = np.corrcoef(our_train[:, elec], original_train[:, elec])[0, 1]
            train_elec_corrs.append(corr)

        if original_test[:, elec].std() > 0:
            corr = np.corrcoef(our_test[:, elec], original_test[:, elec])[0, 1]
            test_elec_corrs.append(corr)

    print(f"\nPer-Electrode Correlation:")
    print(f"  Train - Mean: {np.mean(train_elec_corrs):.6f}, "
          f"Min: {np.min(train_elec_corrs):.6f}, "
          f"% > 0.95: {100*np.mean(np.array(train_elec_corrs) > 0.95):.1f}%")
    print(f"  Test  - Mean: {np.mean(test_elec_corrs):.6f}, "
          f"Min: {np.min(test_elec_corrs):.6f}, "
          f"% > 0.95: {100*np.mean(np.array(test_elec_corrs) > 0.95):.1f}%")

    validation_stats['train_elec_corr_mean'] = np.mean(train_elec_corrs)
    validation_stats['train_elec_corr_min'] = np.min(train_elec_corrs)
    validation_stats['test_elec_corr_mean'] = np.mean(test_elec_corrs)
    validation_stats['test_elec_corr_min'] = np.min(test_elec_corrs)

    # Distribution comparison (KS test)
    ks_train = ks_2samp(our_train.flatten(), original_train.flatten())
    ks_test = ks_2samp(our_test.flatten(), original_test.flatten())

    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Train - statistic: {ks_train.statistic:.6f}, p-value: {ks_train.pvalue:.6f}")
    print(f"  Test  - statistic: {ks_test.statistic:.6f}, p-value: {ks_test.pvalue:.6f}")

    validation_stats['ks_train_stat'] = ks_train.statistic
    validation_stats['ks_train_pval'] = ks_train.pvalue
    validation_stats['ks_test_stat'] = ks_test.statistic
    validation_stats['ks_test_pval'] = ks_test.pvalue

    # MSE and MAE
    train_mse = np.mean((our_train - original_train) ** 2)
    test_mse = np.mean((our_test - original_test) ** 2)
    train_mae = np.mean(np.abs(our_train - original_train))
    test_mae = np.mean(np.abs(our_test - original_test))

    print(f"\nMean Squared Error:")
    print(f"  Train: {train_mse:.6f}")
    print(f"  Test:  {test_mse:.6f}")
    print(f"\nMean Absolute Error:")
    print(f"  Train: {train_mae:.6f}")
    print(f"  Test:  {test_mae:.6f}")

    validation_stats['train_mse'] = train_mse
    validation_stats['test_mse'] = test_mse
    validation_stats['train_mae'] = train_mae
    validation_stats['test_mae'] = test_mae

    # Mean and std comparison
    print(f"\nMean Comparison:")
    print(f"  Train - Ours: {our_train.mean():.6f}, Original: {original_train.mean():.6f}")
    print(f"  Test  - Ours: {our_test.mean():.6f}, Original: {original_test.mean():.6f}")
    print(f"\nStd Comparison:")
    print(f"  Train - Ours: {our_train.std():.6f}, Original: {original_train.std():.6f}")
    print(f"  Test  - Ours: {our_test.std():.6f}, Original: {original_test.std():.6f}")

    # Create validation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot - train
    axes[0, 0].scatter(original_train.flatten(), our_train.flatten(),
                       alpha=0.1, s=1)
    axes[0, 0].plot([original_train.min(), original_train.max()],
                    [original_train.min(), original_train.max()],
                    'r--', linewidth=2, label='Identity')
    axes[0, 0].set_xlabel('Original Train MUA')
    axes[0, 0].set_ylabel('Our Train MUA')
    axes[0, 0].set_title(f'Train Data (corr={train_corr:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot - test
    axes[0, 1].scatter(original_test.flatten(), our_test.flatten(),
                       alpha=0.1, s=1)
    axes[0, 1].plot([original_test.min(), original_test.max()],
                    [original_test.min(), original_test.max()],
                    'r--', linewidth=2, label='Identity')
    axes[0, 1].set_xlabel('Original Test MUA')
    axes[0, 1].set_ylabel('Our Test MUA')
    axes[0, 1].set_title(f'Test Data (corr={test_corr:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram comparison - train
    axes[1, 0].hist(original_train.flatten(), bins=100, alpha=0.5,
                    label='Original', density=True)
    axes[1, 0].hist(our_train.flatten(), bins=100, alpha=0.5,
                    label='Ours', density=True)
    axes[1, 0].set_xlabel('Train MUA Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Train Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram comparison - test
    axes[1, 1].hist(original_test.flatten(), bins=100, alpha=0.5,
                    label='Original', density=True)
    axes[1, 1].hist(our_test.flatten(), bins=100, alpha=0.5,
                    label='Ours', density=True)
    axes[1, 1].set_xlabel('Test MUA Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Test Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{monkey_name}_validation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved validation plots to {plot_path}")
    plt.close()

    print("\n" + "="*60)

    return validation_stats


def main():
    parser = argparse.ArgumentParser(
        description='Verify paper normalization method against THINGS_normMUA.mat'
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

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    allmua, allmat = load_mua_data(args.data_dir, args.monkey)

    # Initialize normalizer
    normalizer = PaperNormalization(args.monkey)

    # Normalize
    print("\nApplying paper's normalization method...")
    train_MUA, test_MUA, test_MUA_reps, stats = normalizer.fit_transform(allmua, allmat)

    print(f"\nResults:")
    print(f"  train_MUA shape: {train_MUA.shape}")
    print(f"  test_MUA shape: {test_MUA.shape}")
    print(f"  test_MUA_reps shape: {test_MUA_reps.shape}")

    # Save normalized data
    output_file = os.path.join(args.output_dir, f'{args.monkey}_paper_normalized.mat')
    savemat(output_file, {
        'train_MUA': train_MUA,
        'test_MUA': test_MUA,
        'test_MUA_reps': test_MUA_reps
    })
    print(f"\nSaved normalized data to {output_file}")

    # Load original normalized data for validation
    try:
        original_file = os.path.join(args.data_dir, f'{args.monkey}_THINGS_normMUA.mat')
        original_data = load_mat_file(original_file)

        original_train = original_data['train_MUA']
        original_test = original_data['test_MUA']

        print(f"\nLoaded original data for validation:")
        print(f"  Original train_MUA shape: {original_train.shape}")
        print(f"  Original test_MUA shape: {original_test.shape}")

        # Validate
        validation_stats = validate_reconstruction(
            train_MUA, test_MUA,
            original_train, original_test,
            args.monkey, args.output_dir
        )

        # Save validation report
        report_file = os.path.join(args.output_dir, f'{args.monkey}_validation_report.txt')
        with open(report_file, 'w') as f:
            f.write("PAPER NORMALIZATION VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Monkey: {args.monkey}\n\n")

            f.write("Normalization Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nValidation Statistics:\n")
            for key, value in validation_stats.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nValidation Summary:\n")
            train_corr = validation_stats['train_global_corr']
            test_corr = validation_stats['test_global_corr']

            if train_corr > 0.99 and test_corr > 0.99:
                f.write("  ✓ PASS - Excellent match (correlation > 0.99)\n")
            elif train_corr > 0.95 and test_corr > 0.95:
                f.write("  ✓ PASS - Good match (correlation > 0.95)\n")
            else:
                f.write("  ✗ FAIL - Poor match (correlation < 0.95)\n")
                f.write("  ACTION REQUIRED: Review data structure and normalization logic\n")

        print(f"Saved validation report to {report_file}")

    except FileNotFoundError:
        print(f"\nWarning: Could not find {original_file}")
        print("Skipping validation against original data")


if __name__ == '__main__':
    main()
