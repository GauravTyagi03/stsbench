"""
Visualize Normalization Stages

This script visualizes raw, baseline-normalized, and fully-normalized data
to understand the effect of each normalization stage.

Usage:
    python visualize_normalization.py --monkey monkeyF
    python visualize_normalization.py --monkey monkeyN --trial_idx 100 --electrode_idx 600
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
import os
import sys
import json
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_raw_data(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw MUA data.

    Args:
        data_dir: Path to data directory
        monkey_name: 'monkeyF' or 'monkeyN'

    Returns:
        allmua: (n_timepoints, n_electrodes, n_trials)
        allmat: (6, n_trials)
    """
    print(f"Loading raw data for {monkey_name}...")

    # Load main data file
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')

    with h5py.File(data_file, 'r') as f:
        allmua = np.array(f['ALLMUA'])
        allmat = np.array(f['ALLMAT'])

    # Load and apply channel mapping - flexible search for any mapping file
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    if not mapping_files:
        raise FileNotFoundError(f"Could not find mapping file for {monkey_name} in {data_dir}")

    mapping_file = os.path.join(data_dir, mapping_files[0])
    print(f"  Using mapping file: {mapping_files[0]}")

    mapping_data = loadmat(mapping_file)
    mapping = mapping_data['mapping'].flatten() - 1

    allmua = allmua[..., mapping]

    # h5py loads data as (n_timepoints, n_trials, n_electrodes)
    # Need to transpose to (n_timepoints, n_electrodes, n_trials)
    print(f"  Transposing from {allmua.shape} to (timepoints, electrodes, trials)")
    allmua = np.transpose(allmua, (0, 2, 1))

    print(f"  ALLMUA shape: {allmua.shape}")
    print(f"  ALLMAT shape: {allmat.shape}")

    return allmua, allmat


def load_normalized_data(
    results_dir: str,
    monkey_name: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load normalized data and metadata.

    Args:
        results_dir: Path to results directory
        monkey_name: Monkey name

    Returns:
        baseline_normalized: (n_timepoints, n_electrodes, n_trials)
        final_normalized: (n_bins, n_electrodes, n_trials)
        metadata: Normalization metadata
    """
    print(f"Loading normalized data for {monkey_name}...")

    # Load baseline-normalized data
    baseline_file = os.path.join(results_dir, f'{monkey_name}_baseline_normalized.mat')
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Could not find {baseline_file}")

    baseline_data = loadmat(baseline_file)
    baseline_normalized = baseline_data['baseline_normalized']

    # Load final normalized data
    final_file = os.path.join(results_dir, f'{monkey_name}_timeseries_normalized.mat')
    if not os.path.exists(final_file):
        raise FileNotFoundError(f"Could not find {final_file}")

    final_data = loadmat(final_file)
    final_normalized = final_data['timeseries_normalized']

    # Load metadata
    metadata_file = os.path.join(results_dir, f'{monkey_name}_normalization_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"  Warning: Could not find {metadata_file}, using defaults")
        metadata = {'baseline_window': 100, 'bin_width': 10}

    print(f"  Baseline-normalized shape: {baseline_normalized.shape}")
    print(f"  Final normalized shape: {final_normalized.shape}")

    return baseline_normalized, final_normalized, metadata


def get_trial_metadata(allmat: np.ndarray, trial_idx: int) -> Dict:
    """
    Extract metadata for a specific trial.

    Args:
        allmat: (6, n_trials)
        trial_idx: Trial index

    Returns:
        metadata: Dictionary with trial information
    """
    train_idx = int(allmat[1, trial_idx])
    test_idx = int(allmat[2, trial_idx])
    day = int(allmat[5, trial_idx])

    trial_type = 'train' if train_idx > 0 else 'test'
    stim_id = train_idx if train_idx > 0 else test_idx

    return {
        'trial_type': trial_type,
        'stimulus_id': stim_id,
        'day': day
    }


def get_brain_region(electrode_idx: int, monkey_name: str) -> str:
    """Get brain region for electrode."""
    if monkey_name == 'monkeyF':
        if electrode_idx < 512:
            return 'V1'
        elif electrode_idx < 832:
            return 'IT'
        else:
            return 'V4'
    else:  # monkeyN
        if electrode_idx < 512:
            return 'V1'
        elif electrode_idx < 768:
            return 'V4'
        else:
            return 'IT'


def plot_normalization_stages(
    raw: np.ndarray,
    baseline_norm: np.ndarray,
    full_norm: np.ndarray,
    trial_idx: int,
    electrode_idx: int,
    metadata: Dict,
    trial_metadata: Dict,
    monkey_name: str,
    output_path: str
):
    """
    Plot the three stages of normalization for a single trial.

    Args:
        raw: (n_timepoints, n_electrodes, n_trials)
        baseline_norm: (n_timepoints, n_electrodes, n_trials)
        full_norm: (n_bins, n_electrodes, n_trials)
        trial_idx: Index of trial to plot
        electrode_idx: Index of electrode to plot
        metadata: Normalization metadata
        trial_metadata: Trial-specific metadata
        monkey_name: Monkey name
        output_path: Path to save figure
    """
    baseline_window = metadata.get('baseline_window', 100)
    bin_width = metadata.get('bin_width', 10)
    brain_region = get_brain_region(electrode_idx, monkey_name)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # Time axis for raw and baseline-normalized data
    time_ms = np.arange(raw.shape[0])

    # Row 1: Raw timeseries
    axes[0].plot(time_ms, raw[:, electrode_idx, trial_idx],
                 linewidth=1.5, color='#2E86AB')
    axes[0].axvspan(0, baseline_window, alpha=0.15, color='gray',
                    label=f'Baseline window (0-{baseline_window}ms)')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Raw MUA', fontsize=11)
    axes[0].set_title(
        f'Raw MUA - Trial {trial_idx}, Electrode {electrode_idx} ({brain_region})',
        fontsize=12, fontweight='bold'
    )
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Row 2: Baseline-normalized timeseries
    axes[1].plot(time_ms, baseline_norm[:, electrode_idx, trial_idx],
                 linewidth=1.5, color='#A23B72')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                    label='Zero line')
    axes[1].axvspan(0, baseline_window, alpha=0.15, color='gray')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_ylabel('Baseline-Normalized MUA', fontsize=11)
    axes[1].set_title(
        'After Stage 1: Baseline Normalization',
        fontsize=12, fontweight='bold'
    )
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Row 3: Fully-normalized (binned) timeseries
    n_bins = full_norm.shape[0]
    time_bins = np.arange(n_bins) * bin_width + bin_width / 2  # Bin centers
    axes[2].plot(time_bins, full_norm[:, electrode_idx, trial_idx],
                 linewidth=2, color='#F18F01', marker='o', markersize=4)
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].set_xlabel('Time (ms)', fontsize=11)
    axes[2].set_ylabel('Fully-Normalized MUA', fontsize=11)
    axes[2].set_title(
        f'After Stage 2: Test-Pool Bin Normalization (bin width={bin_width}ms)',
        fontsize=12, fontweight='bold'
    )
    axes[2].grid(True, alpha=0.3)

    # Add trial metadata
    info_text = (
        f"Trial Type: {trial_metadata['trial_type'].upper()}\n"
        f"Stimulus ID: {trial_metadata['stimulus_id']}\n"
        f"Recording Day: {trial_metadata['day']}"
    )
    fig.text(0.99, 0.98, info_text, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved single-trial visualization to {output_path}")
    plt.close()


def plot_multi_trial_overlay(
    raw: np.ndarray,
    baseline_norm: np.ndarray,
    full_norm: np.ndarray,
    electrode_idx: int,
    trial_indices: np.ndarray,
    metadata: Dict,
    allmat: np.ndarray,
    monkey_name: str,
    output_path: str
):
    """
    Plot multiple trials overlaid to show variance reduction.

    Args:
        raw: (n_timepoints, n_electrodes, n_trials)
        baseline_norm: (n_timepoints, n_electrodes, n_trials)
        full_norm: (n_bins, n_electrodes, n_trials)
        electrode_idx: Index of electrode to plot
        trial_indices: Array of trial indices to plot
        metadata: Normalization metadata
        allmat: (6, n_trials)
        monkey_name: Monkey name
        output_path: Path to save figure
    """
    baseline_window = metadata.get('baseline_window', 100)
    bin_width = metadata.get('bin_width', 10)
    brain_region = get_brain_region(electrode_idx, monkey_name)

    n_trials = len(trial_indices)
    colors = plt.cm.viridis(np.linspace(0, 1, n_trials))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    time_ms = np.arange(raw.shape[0])

    # Row 1: Raw data overlay
    for i, trial_idx in enumerate(trial_indices):
        axes[0].plot(time_ms, raw[:, electrode_idx, trial_idx],
                     alpha=0.6, linewidth=1, color=colors[i])
    axes[0].axvspan(0, baseline_window, alpha=0.15, color='gray',
                    label=f'Baseline window')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Raw MUA', fontsize=11)
    axes[0].set_title(
        f'Raw MUA - {n_trials} Trials, Electrode {electrode_idx} ({brain_region})',
        fontsize=12, fontweight='bold'
    )
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Row 2: Baseline-normalized overlay
    for i, trial_idx in enumerate(trial_indices):
        axes[1].plot(time_ms, baseline_norm[:, electrode_idx, trial_idx],
                     alpha=0.6, linewidth=1, color=colors[i])
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[1].axvspan(0, baseline_window, alpha=0.15, color='gray')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_ylabel('Baseline-Normalized MUA', fontsize=11)
    axes[1].set_title(
        'After Stage 1: Baseline Normalization',
        fontsize=12, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3)

    # Row 3: Fully-normalized overlay
    n_bins = full_norm.shape[0]
    time_bins = np.arange(n_bins) * bin_width + bin_width / 2
    for i, trial_idx in enumerate(trial_indices):
        axes[2].plot(time_bins, full_norm[:, electrode_idx, trial_idx],
                     alpha=0.7, linewidth=1.5, color=colors[i],
                     marker='o', markersize=3)
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].set_xlabel('Time (ms)', fontsize=11)
    axes[2].set_ylabel('Fully-Normalized MUA', fontsize=11)
    axes[2].set_title(
        f'After Stage 2: Test-Pool Bin Normalization',
        fontsize=12, fontweight='bold'
    )
    axes[2].grid(True, alpha=0.3)

    # Compute and display variance reduction
    raw_var = np.var([raw[:, electrode_idx, idx] for idx in trial_indices])
    baseline_var = np.var([baseline_norm[:, electrode_idx, idx] for idx in trial_indices])
    final_var = np.var([full_norm[:, electrode_idx, idx] for idx in trial_indices])

    variance_text = (
        f"Variance across trials:\n"
        f"Raw: {raw_var:.4f}\n"
        f"Baseline-norm: {baseline_var:.4f}\n"
        f"Final: {final_var:.4f}"
    )
    fig.text(0.99, 0.98, variance_text, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round',
             facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-trial visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize normalization stages'
    )
    parser.add_argument('--monkey', type=str, required=True,
                        choices=['monkeyF', 'monkeyN'],
                        help='Monkey name')
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/groups/anishm/tvsd/',
                        help='Path to raw data directory')
    parser.add_argument('--results_dir', type=str,
                        default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/',
                        help='Path to normalized data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/',
                        help='Path to save plots')
    parser.add_argument('--trial_idx', type=int, default=None,
                        help='Specific trial index (default: random)')
    parser.add_argument('--electrode_idx', type=int, default=None,
                        help='Specific electrode index (default: random from V4)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials for multi-trial plot')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("NORMALIZATION VISUALIZATION")
    print("="*60)
    print(f"Monkey: {args.monkey}")
    print("="*60)

    # Load data
    raw, allmat = load_raw_data(args.data_dir, args.monkey)
    baseline_norm, full_norm, metadata = load_normalized_data(
        args.results_dir, args.monkey
    )

    n_timepoints, n_electrodes, n_trials = raw.shape

    # Select electrode (default: random from V4)
    if args.electrode_idx is None:
        if args.monkey == 'monkeyF':
            v4_range = range(832, 1024)
        else:
            v4_range = range(512, 768)
        electrode_idx = np.random.choice(v4_range)
        print(f"\nRandomly selected electrode: {electrode_idx} (V4)")
    else:
        electrode_idx = min(max(0, args.electrode_idx), n_electrodes - 1)
        print(f"\nUsing specified electrode: {electrode_idx}")

    # Select trial for single-trial plot
    if args.trial_idx is None:
        trial_idx = np.random.randint(0, n_trials)
        print(f"Randomly selected trial: {trial_idx}")
    else:
        trial_idx = min(max(0, args.trial_idx), n_trials - 1)
        print(f"Using specified trial: {trial_idx}")

    trial_metadata = get_trial_metadata(allmat, trial_idx)
    print(f"  Trial type: {trial_metadata['trial_type']}")
    print(f"  Stimulus ID: {trial_metadata['stimulus_id']}")
    print(f"  Day: {trial_metadata['day']}")

    # Single-trial visualization
    print("\nGenerating single-trial visualization...")
    single_output = os.path.join(
        args.output_dir,
        f'{args.monkey}_normalization_single_trial.png'
    )
    plot_normalization_stages(
        raw, baseline_norm, full_norm,
        trial_idx, electrode_idx,
        metadata, trial_metadata,
        args.monkey, single_output
    )

    # Multi-trial visualization
    print("\nGenerating multi-trial overlay visualization...")
    # Select random trials from same stimulus if possible
    if trial_metadata['trial_type'] == 'train':
        same_stim_mask = (allmat[1, :] == trial_metadata['stimulus_id'])
    else:
        same_stim_mask = (allmat[2, :] == trial_metadata['stimulus_id'])

    same_stim_trials = np.where(same_stim_mask)[0]

    if len(same_stim_trials) >= args.n_trials:
        print(f"  Using {args.n_trials} trials from same stimulus")
        trial_indices = np.random.choice(same_stim_trials, args.n_trials, replace=False)
    else:
        print(f"  Using {args.n_trials} random trials")
        trial_indices = np.random.choice(n_trials, args.n_trials, replace=False)

    multi_output = os.path.join(
        args.output_dir,
        f'{args.monkey}_normalization_multi_trial.png'
    )
    plot_multi_trial_overlay(
        raw, baseline_norm, full_norm,
        electrode_idx, trial_indices,
        metadata, allmat,
        args.monkey, multi_output
    )

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output files:")
    print(f"  {single_output}")
    print(f"  {multi_output}")
    print("="*60)


if __name__ == '__main__':
    main()
