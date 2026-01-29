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
            allmua: (n_timepoints, n_trials, n_electrodes)  # CORRECTED from (n_timepoints, n_electrodes, n_trials)
            electrode_idx: Index of electrode

        Returns:
            (n_trials,) averaged response in time window
        """
        region = self._get_brain_region(electrode_idx)
        start, end = self.brain_regions[region]['time_window']

        # CRITICAL FIX: Corrected indexing for shape (timepoints, trials, electrodes)
        # OLD (assumed transposed shape):
        # window_data = allmua[start:end, electrode_idx, :]

        # NEW (for shape (timepoints, trials, electrodes)):
        window_data = allmua[start:end, :, electrode_idx]
        return window_data.mean(axis=0)

    def normalize_electrode_day(
        self,
        allmua: np.ndarray,
        allmat: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize each electrode separately for each day using test pool statistics.

        Args:
            allmua: (n_timepoints, n_trials, n_electrodes)  # CORRECTED from (n_timepoints, n_electrodes, n_trials)
            allmat: (6, n_trials) metadata array

        Returns:
            normalized: (n_electrodes, n_trials)
            stats: Dictionary with normalization statistics
        """
        # CORRECTED: Extract dimensions from corrected shape
        n_electrodes = allmua.shape[2]  # Changed from shape[1]
        n_trials = allmua.shape[1]      # Changed from shape[2]

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
                std = test_data.std(ddof=1)  # Use sample std for unbiased estimation

                if std == 0 or np.isnan(std):
                    std = 1.0
                    stats['zero_std'] += 1

                # Normalize ALL trials in this day using test pool statistics
                normalized[elec_idx, day_mask] = (elec_data[day_mask] - mean) / std

        print(f"Normalization complete. Stats: {stats}")
        return normalized, stats

    def _average_by_stim_ids(
        self,
        normalized: np.ndarray,
        idx_array: np.ndarray,
        unique_ids: np.ndarray
    ) -> np.ndarray:
        """Helper to average normalized data by stimulus IDs."""
        n_stim = len(unique_ids)
        n_electrodes = normalized.shape[0]
        result = np.zeros((n_stim, n_electrodes))
        for i, stim_id in enumerate(unique_ids):
            stim_trials = (idx_array == stim_id)
            result[i, :] = normalized[:, stim_trials].mean(axis=1)
        return result

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

        # Average train stimuli
        train_MUA = self._average_by_stim_ids(normalized, train_idx, unique_train)

        # Average test stimuli
        test_MUA = self._average_by_stim_ids(normalized, test_idx, unique_test)

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
            allmua: (n_timepoints, n_trials, n_electrodes)  # CORRECTED from (n_timepoints, n_electrodes, n_trials)
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
        allmua: (n_timepoints, n_trials, n_electrodes)  # CORRECTED from (n_timepoints, n_electrodes, n_trials)
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
    # CRITICAL FIX: Keep shape as-is - DO NOT transpose
    # The original code transposed to (timepoints, electrodes, trials) but this broke
    # all downstream indexing. The correct shape is (timepoints, trials, electrodes).
    #
    # COMMENTED OUT - transpose was breaking indexing:
    # print(f"Transposing from {allmua.shape} to (timepoints, electrodes, trials)")
    # allmua = np.transpose(allmua, (0, 2, 1))
    

    # Keep shape as (300, 25248, 1024) = (timepoints, trials, electrodes)
    print(f"Final ALLMUA shape (timepoints, trials, electrodes): {allmua.shape}")

    # Sanity check: verify shape is correct
    assert allmua.shape[0] == 300, f"Expected 300 timepoints, got {allmua.shape[0]}"
    assert allmua.shape[2] == 1024, f"Expected 1024 electrodes, got {allmua.shape[2]}"
    print(f"✓ Shape verified: (timepoints={allmua.shape[0]}, trials={allmua.shape[1]}, electrodes={allmua.shape[2]})")

    return allmua, allmat


def validate_distributional_statistics(
    our_train: np.ndarray,
    our_test: np.ndarray,
    original_train: np.ndarray,
    original_test: np.ndarray,
    brain_regions: Dict[str, Dict],
    monkey_name: str,
    output_dir: str
) -> Dict:
    """
    Validate normalization by comparing distributional statistics by brain region.

    This is an order-independent validation that compares mean, std, and distributions
    across brain regions, rather than point-by-point correlations.

    Args:
        our_train: Our normalized training data (n_train_stimuli, n_electrodes)
        our_test: Our normalized test data (n_test_stimuli, n_electrodes)
        original_train: Original training data from THINGS_normMUA.mat
        original_test: Original test data from THINGS_normMUA.mat
        brain_regions: Dict mapping region names to electrode ranges
        monkey_name: Monkey name for output files
        output_dir: Directory for saving results

    Returns:
        validation_stats: Dictionary with validation metrics by region
    """
    print("\n" + "="*60)
    print("DISTRIBUTIONAL STATISTICS VALIDATION")
    print("="*60)

    # Shape validation
    print(f"\nShape Validation:")
    print(f"  Our train shape: {our_train.shape}, Original train shape: {original_train.shape}")
    print(f"  Our test shape: {our_test.shape}, Original test shape: {original_test.shape}")

    if our_train.shape != original_train.shape or our_test.shape != original_test.shape:
        print("  ERROR: Shape mismatch detected!")
        return {}

    # Check for NaN/Inf values
    print(f"\nData Quality Checks:")
    print(f"  Train - Our NaN/Inf: {np.isnan(our_train).sum()}/{np.isinf(our_train).sum()}, "
          f"Original NaN/Inf: {np.isnan(original_train).sum()}/{np.isinf(original_train).sum()}")
    print(f"  Test  - Our NaN/Inf: {np.isnan(our_test).sum()}/{np.isinf(our_test).sum()}, "
          f"Original NaN/Inf: {np.isnan(original_test).sum()}/{np.isinf(original_test).sum()}")

    # Add "ALL" region (all electrodes)
    regions_with_all = dict(brain_regions)
    regions_with_all['ALL'] = {'electrodes': range(our_train.shape[1])}

    # Compute statistics for each region and dataset
    results = {'train': {}, 'test': {}}

    for dataset_name, our_data, orig_data in [
        ('train', our_train, original_train),
        ('test', our_test, original_test)
    ]:
        print(f"\n{dataset_name.upper()} Dataset Statistics:")
        print("-" * 60)

        for region_name, region_info in regions_with_all.items():
            region_electrodes = list(region_info['electrodes'])

            # Extract region data
            region_our = our_data[:, region_electrodes]
            region_orig = orig_data[:, region_electrodes]

            # A. Global statistics (all values in region)
            region_our_flat = region_our.flatten()
            region_orig_flat = region_orig.flatten()

            our_mean = region_our_flat.mean()
            our_std = region_our_flat.std(ddof=1)
            orig_mean = region_orig_flat.mean()
            orig_std = region_orig_flat.std(ddof=1)

            # B. Per-stimulus statistics (variation across stimuli)
            per_stim_our = region_our.mean(axis=1)  # (n_stimuli,)
            per_stim_orig = region_orig.mean(axis=1)

            our_stim_variation = per_stim_our.std(ddof=1)
            orig_stim_variation = per_stim_orig.std(ddof=1)

            # C. Per-electrode statistics (variation across electrodes)
            per_elec_our_mean = region_our.mean(axis=0)  # (n_electrodes_in_region,)
            per_elec_our_std = region_our.std(axis=0, ddof=1)
            per_elec_orig_mean = region_orig.mean(axis=0)
            per_elec_orig_std = region_orig.std(axis=0, ddof=1)

            electrode_mean_variation_our = per_elec_our_mean.std(ddof=1)
            electrode_std_variation_our = per_elec_our_std.std(ddof=1)
            electrode_mean_variation_orig = per_elec_orig_mean.std(ddof=1)
            electrode_std_variation_orig = per_elec_orig_std.std(ddof=1)

            # KS test for distribution comparison
            ks_result = ks_2samp(region_our_flat, region_orig_flat)

            # Store results
            results[dataset_name][region_name] = {
                'our_mean': our_mean,
                'our_std': our_std,
                'original_mean': orig_mean,
                'original_std': orig_std,
                'mean_diff': abs(our_mean - orig_mean),
                'std_diff': abs(our_std - orig_std),
                'std_ratio': our_std / orig_std if orig_std > 0 else np.nan,
                'our_stim_variation': our_stim_variation,
                'original_stim_variation': orig_stim_variation,
                'electrode_mean_variation_our': electrode_mean_variation_our,
                'electrode_std_variation_our': electrode_std_variation_our,
                'electrode_mean_variation_orig': electrode_mean_variation_orig,
                'electrode_std_variation_orig': electrode_std_variation_orig,
                'ks_statistic': ks_result.statistic,
                'ks_pvalue': ks_result.pvalue,
                'n_electrodes': len(region_electrodes)
            }

            # Print summary
            stats = results[dataset_name][region_name]
            print(f"\n{region_name} ({stats['n_electrodes']} electrodes):")
            print(f"  Mean:  Ours={our_mean:.4f}, Orig={orig_mean:.4f}, Diff={stats['mean_diff']:.4f}")
            print(f"  Std:   Ours={our_std:.4f}, Orig={orig_std:.4f}, Ratio={stats['std_ratio']:.4f}")
            print(f"  Stim Variation: Ours={our_stim_variation:.4f}, Orig={orig_stim_variation:.4f}")
            print(f"  KS Test: stat={ks_result.statistic:.4f}, p={ks_result.pvalue:.4f}")

    # Create visualizations
    _create_validation_plots(results, our_train, our_test, original_train, original_test,
                              regions_with_all, monkey_name, output_dir)

    # Create summary report
    _create_validation_report(results, monkey_name, output_dir)

    print("\n" + "="*60)

    return results


def _create_validation_plots(
    results: Dict,
    our_train: np.ndarray,
    our_test: np.ndarray,
    original_train: np.ndarray,
    original_test: np.ndarray,
    brain_regions: Dict[str, Dict],
    monkey_name: str,
    output_dir: str
):
    """Create comprehensive validation plots."""

    # Create 2 figures (one for train, one for test)
    for dataset_name, our_data, orig_data in [
        ('train', our_train, original_train),
        ('test', our_test, original_test)
    ]:
        # We plot 7 panels total: 3 summary bar charts + 4 region histograms.
        # Use a 2x4 grid (8 slots) to avoid subplot indexing errors.
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Panel 1: Mean Comparison by Region
        ax1 = axes[0, 0]
        regions = list(results[dataset_name].keys())
        our_means = [results[dataset_name][r]['our_mean'] for r in regions]
        orig_means = [results[dataset_name][r]['original_mean'] for r in regions]

        x = np.arange(len(regions))
        width = 0.35
        ax1.bar(x - width/2, orig_means, width, label='Original', alpha=0.8)
        ax1.bar(x + width/2, our_means, width, label='Ours', alpha=0.8)
        ax1.set_xlabel('Brain Region')
        ax1.set_ylabel('Mean')
        ax1.set_title('Mean Comparison by Region')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Std Comparison by Region
        ax2 = axes[0, 1]
        our_stds = [results[dataset_name][r]['our_std'] for r in regions]
        orig_stds = [results[dataset_name][r]['original_std'] for r in regions]

        ax2.bar(x - width/2, orig_stds, width, label='Original', alpha=0.8)
        ax2.bar(x + width/2, our_stds, width, label='Ours', alpha=0.8)
        ax2.set_xlabel('Brain Region')
        ax2.set_ylabel('Std')
        ax2.set_title('Std Comparison by Region')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Per-Stimulus Variation
        ax3 = axes[0, 2]
        our_stim_vars = [results[dataset_name][r]['our_stim_variation'] for r in regions]
        orig_stim_vars = [results[dataset_name][r]['original_stim_variation'] for r in regions]

        ax3.bar(x - width/2, orig_stim_vars, width, label='Original', alpha=0.8)
        ax3.bar(x + width/2, our_stim_vars, width, label='Ours', alpha=0.8)
        ax3.set_xlabel('Brain Region')
        ax3.set_ylabel('Stimulus Variation (Std)')
        ax3.set_title('Per-Stimulus Variation by Region')
        ax3.set_xticks(x)
        ax3.set_xticklabels(regions)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: unused (kept for layout symmetry)
        axes[0, 3].axis('off')

        # Panel 4-7: Distribution Histograms by Region (4 subplots)
        for idx, region_name in enumerate(['V1', 'V4', 'IT', 'ALL']):
            ax = axes[1, idx]
            region_electrodes = list(brain_regions[region_name]['electrodes'])

            region_our = our_data[:, region_electrodes].flatten()
            region_orig = orig_data[:, region_electrodes].flatten()

            # For training data, clip to [-10, 10] to get smaller bins
            if dataset_name == 'train':
                region_our_clipped = np.clip(region_our, -10, 10)
                region_orig_clipped = np.clip(region_orig, -10, 10)
                bins = np.linspace(-10, 10, 100)  # 100 bins over [-10, 10] range
            else:
                region_our_clipped = region_our
                region_orig_clipped = region_orig
                bins = 50

            ax.hist(region_orig_clipped, bins=bins, alpha=0.5, label='Original', density=True)
            ax.hist(region_our_clipped, bins=bins, alpha=0.5, label='Ours', density=True)
            ax.set_xlabel('MUA Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{region_name} Distribution')
            ax.set_xlim(-3, 3)  # Clip x-axis to [-3, 3]
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'{dataset_name.upper()} Dataset - Distributional Validation',
            fontsize=14,
            fontweight='bold'
        )
        # Leave space for the suptitle.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(output_dir, f'{monkey_name}_{dataset_name}_distributional_validation.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved {dataset_name} validation plot to {plot_path}")
        plt.close()


def _create_validation_report(results: Dict, monkey_name: str, output_dir: str):
    """Create detailed validation report."""

    report_path = os.path.join(output_dir, f'{monkey_name}_distributional_validation_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISTRIBUTIONAL STATISTICS VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Monkey: {monkey_name}\n\n")

        # Validation criteria
        f.write("VALIDATION CRITERIA:\n")
        f.write("  - Mean difference < 0.01 per region\n")
        f.write("  - Std ratio within [0.95, 1.05] per region\n")
        f.write("  - KS test p-value > 0.05 per region\n\n")

        for dataset_name in ['train', 'test']:
            f.write("="*80 + "\n")
            f.write(f"{dataset_name.upper()} DATASET\n")
            f.write("="*80 + "\n\n")

            # Table header
            f.write(f"{'Region':<8} {'N_Elec':<8} {'Our_Mean':<10} {'Orig_Mean':<10} {'Mean_Diff':<10} "
                   f"{'Our_Std':<10} {'Orig_Std':<10} {'Std_Ratio':<10} {'KS_pval':<10} {'Pass':<5}\n")
            f.write("-"*100 + "\n")

            # Table rows
            all_pass = True
            for region_name in ['V1', 'V4', 'IT', 'ALL']:
                stats = results[dataset_name][region_name]

                # Check pass criteria
                mean_pass = stats['mean_diff'] < 0.01
                std_pass = 0.95 <= stats['std_ratio'] <= 1.05
                ks_pass = stats['ks_pvalue'] > 0.05
                region_pass = mean_pass and std_pass and ks_pass
                all_pass = all_pass and region_pass

                f.write(f"{region_name:<8} {stats['n_electrodes']:<8} "
                       f"{stats['our_mean']:<10.4f} {stats['original_mean']:<10.4f} {stats['mean_diff']:<10.4f} "
                       f"{stats['our_std']:<10.4f} {stats['original_std']:<10.4f} {stats['std_ratio']:<10.4f} "
                       f"{stats['ks_pvalue']:<10.4f} {'PASS' if region_pass else 'FAIL':<5}\n")

            f.write("\n")

            # Investigation of training std
            if dataset_name == 'train':
                f.write("INVESTIGATION OF TRAINING STD:\n")
                f.write("-"*80 + "\n")

                train_all_std = results['train']['ALL']['our_std']
                test_all_std = results['test']['ALL']['our_std']

                f.write(f"\nTrain vs Test Std Comparison:\n")
                f.write(f"  Train Std: {train_all_std:.4f}\n")
                f.write(f"  Test Std:  {test_all_std:.4f}\n")
                f.write(f"  Ratio (test/train): {test_all_std/train_all_std:.4f}\n")

                f.write(f"\nPer-Stimulus Variation (std of stimulus means):\n")
                for region in ['V1', 'V4', 'IT', 'ALL']:
                    train_stim_var = results['train'][region]['our_stim_variation']
                    test_stim_var = results['test'][region]['our_stim_variation']
                    f.write(f"  {region}: Train={train_stim_var:.4f}, Test={test_stim_var:.4f}\n")

                f.write(f"\nPer-Electrode Variation:\n")
                for region in ['V1', 'V4', 'IT', 'ALL']:
                    elec_mean_var = results['train'][region]['electrode_mean_variation_our']
                    elec_std_var = results['train'][region]['electrode_std_variation_our']
                    f.write(f"  {region}: Mean_Var={elec_mean_var:.4f}, Std_Var={elec_std_var:.4f}\n")

                f.write("\n")

            # Summary
            f.write(f"\n{dataset_name.upper()} SUMMARY: ")
            if all_pass:
                f.write("PASS - All regions meet validation criteria\n\n")
            else:
                f.write("FAIL - Some regions do not meet validation criteria\n\n")

        # Overall summary
        train_pass = all(
            results['train'][r]['mean_diff'] < 0.01 and
            0.95 <= results['train'][r]['std_ratio'] <= 1.05 and
            results['train'][r]['ks_pvalue'] > 0.05
            for r in ['V1', 'V4', 'IT', 'ALL']
        )
        test_pass = all(
            results['test'][r]['mean_diff'] < 0.01 and
            0.95 <= results['test'][r]['std_ratio'] <= 1.05 and
            results['test'][r]['ks_pvalue'] > 0.05
            for r in ['V1', 'V4', 'IT', 'ALL']
        )

        f.write("="*80 + "\n")
        f.write("OVERALL VALIDATION: ")
        if train_pass and test_pass:
            f.write("PASS\n")
        else:
            f.write("FAIL\n")
        f.write("="*80 + "\n")

    print(f"Saved validation report to {report_path}")


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

        # Validate using distributional statistics
        validation_results = validate_distributional_statistics(
            train_MUA, test_MUA,
            original_train, original_test,
            normalizer.brain_regions,
            args.monkey, args.output_dir
        )

    except FileNotFoundError:
        print(f"\nWarning: Could not find {original_file}")
        print("Skipping validation against original data")


if __name__ == '__main__':
    main()
