"""
Visualize Normalization Results

Compares raw MUA data with the final normalized output from alternative_timeseries_norm.py
Creates 5 separate plots comparing trials from different days/regions.
All plots now show EVERY single datapoint (no binning/sampling).

Usage:
    python visualize_normalization.py --monkey monkeyF
    python visualize_normalization.py --monkey monkeyN --trial1_idx 100 --trial2_idx 500
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alternative_timeseries_norm import load_normalized_h5


def load_data(data_dir: str, results_dir: str, monkey_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw and normalized data."""
    print(f"Loading data for {monkey_name}...")

    # Load raw data
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')
    with h5py.File(data_file, 'r') as f:
        raw = np.array(f['ALLMUA'])
        allmat = np.array(f['ALLMAT'])
        tb = np.array(f['tb']).flatten()

    # Apply channel mapping
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    if not mapping_files:
        raise FileNotFoundError(f"No mapping file found for {monkey_name}")

    mapping_data = loadmat(os.path.join(data_dir, mapping_files[0]))
    mapping = mapping_data['mapping'].flatten() - 1
    raw = raw[..., mapping]
    raw = np.transpose(raw, (0, 2, 1))  # (timepoints, electrodes, trials)

    # Load normalized data (prefer HDF5 for partial loading; fall back to .mat)
    norm_file_h5 = os.path.join(results_dir, f'{monkey_name}_timeseries_normalized.h5')
    norm_file_mat = os.path.join(results_dir, f'{monkey_name}_timeseries_normalized.mat')
    if os.path.exists(norm_file_h5):
        normalized = load_normalized_h5(norm_file_h5, key='timeseries_normalized')
        # Load time base from normalized file
        with h5py.File(norm_file_h5, 'r') as f:
            if 'tb' in f:
                tb_norm = f['tb'][:]
            else:
                # Fallback: truncate original tb to match normalized length
                tb_norm = tb[:normalized.shape[0]]
                print("  Warning: Time base not found in normalized file, using truncated raw tb")
    elif os.path.exists(norm_file_mat):
        norm_data = loadmat(norm_file_mat)
        normalized = norm_data['timeseries_normalized']
        # Fallback: truncate original tb to match normalized length
        tb_norm = tb[:normalized.shape[0]]
        print("  Warning: Using .mat file, truncating raw tb to match normalized length")
    else:
        raise FileNotFoundError(
            f"Normalized data not found. Looked for: {norm_file_h5!r}, {norm_file_mat!r}"
        )

    print(f"  Raw: {raw.shape}, Normalized: {normalized.shape}")
    print(f"  Raw time base: {tb.min():.1f} to {tb.max():.1f} ms ({len(tb)} timepoints)")
    print(f"  Normalized time base: {tb_norm.min():.1f} to {tb_norm.max():.1f} ms ({len(tb_norm)} timepoints)")

    return raw, normalized, allmat, tb, tb_norm


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


def select_diverse_trials(allmat: np.ndarray, n_electrodes: int, monkey_name: str, seed: int = 42) -> Tuple:
    """
    Intelligently select 2 trials from different days and regions.

    Returns:
        trial1_idx, trial2_idx, electrode1_idx, electrode2_idx
    """
    np.random.seed(seed)

    days = allmat[5].astype(int)
    unique_days = np.unique(days)

    # Define region ranges
    if monkey_name == 'monkeyF':
        regions = {'V1': range(0, 512), 'IT': range(512, 832), 'V4': range(832, 1024)}
    else:
        regions = {'V1': range(0, 512), 'V4': range(512, 768), 'IT': range(768, 1024)}

    region_names = list(regions.keys())

    # Select 2 different days
    if len(unique_days) >= 2:
        day1, day2 = np.random.choice(unique_days, 2, replace=False)
    else:
        day1 = day2 = unique_days[0]
        print("  Warning: Only one day available, using same day for both trials")

    # Select 2 different regions
    region1_name, region2_name = np.random.choice(region_names, 2, replace=False)

    # Find trials from day1 and day2
    day1_trials = np.where(days == day1)[0]
    day2_trials = np.where(days == day2)[0]

    # Select random trials
    trial1_idx = np.random.choice(day1_trials)
    trial2_idx = np.random.choice(day2_trials)

    # Select random electrodes from each region
    electrode1_idx = np.random.choice(regions[region1_name])
    electrode2_idx = np.random.choice(regions[region2_name])

    print(f"\nSelected Trial 1: trial={trial1_idx}, electrode={electrode1_idx} ({region1_name}), day={day1}")
    print(f"Selected Trial 2: trial={trial2_idx}, electrode={electrode2_idx} ({region2_name}), day={day2}")

    return trial1_idx, trial2_idx, electrode1_idx, electrode2_idx


def plot_single_raw_vs_norm(tb, time_bins, raw1, norm1, region1, day1, electrode1_idx, monkey_name, output_path):
    """Plot 1: Single trial raw vs normalized."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(tb, raw1, label='Raw', linewidth=1.5, color='#2E86AB', alpha=0.7)
    ax.plot(time_bins, norm1, label='Normalized', linewidth=2, color='#F18F01')
    ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Stimulus onset')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('MUA', fontsize=11)
    ax.set_title(f'{monkey_name} - Trial 1: Raw vs Normalized\n({region1}, Day {day1}, Electrode {electrode1_idx})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_two_trials_norm(time_bins, norm1, norm2, region1, region2, day1, day2, monkey_name, output_path):
    """Plot 2: Two trials normalized only (different days/regions)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time_bins, norm1, label=f'Trial 1 ({region1}, Day {day1})',
            linewidth=2, color='#A23B72')
    ax.plot(time_bins, norm2, label=f'Trial 2 ({region2}, Day {day2})',
            linewidth=2, color='#17BEBB')
    ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Stimulus onset')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Normalized MUA', fontsize=11)
    ax.set_title(f'{monkey_name} - Two Trials: Normalized Only\n(Different Days/Regions)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_four(tb, time_bins, raw1, raw2, norm1, norm2, region1, region2, monkey_name, output_path):
    """Plot 3: All four together (raw1, norm1, raw2, norm2)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(tb, raw1, label=f'Raw 1 ({region1})',
            linewidth=1.5, color='#2E86AB', alpha=0.5, linestyle='-')
    ax.plot(tb, raw2, label=f'Raw 2 ({region2})',
            linewidth=1.5, color='#006D77', alpha=0.5, linestyle='-')
    ax.plot(time_bins, norm1, label=f'Norm 1 ({region1})',
            linewidth=2, color='#A23B72')
    ax.plot(time_bins, norm2, label=f'Norm 2 ({region2})',
            linewidth=2, color='#17BEBB')
    ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('MUA', fontsize=11)
    ax.set_title(f'{monkey_name} - All Four Together\n(Raw + Normalized for Both Trials)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_single_vs_average(time_bins, norm1, norm_avg_all, region1, day1, trial1_idx, electrode1_idx, monkey_name, n_trials, output_path):
    """Plot 4: Single trial vs average across ALL trials (same electrode)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time_bins, norm1, label=f'Single Trial (Trial {trial1_idx}, Day {day1})',
            linewidth=2.5, color='#A23B72', alpha=0.7)
    ax.plot(time_bins, norm_avg_all, label=f'Average (all {n_trials} trials)',
            linewidth=3, color='#F18F01')
    ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Stimulus onset')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Normalized MUA', fontsize=11)
    ax.set_title(f'{monkey_name} - Single Trial vs Population Average\n(Electrode {electrode1_idx}, {region1})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (
        f"Single trial: mean={norm1.mean():.3f}, std={norm1.std():.3f}\n"
        f"Average (all trials): mean={norm_avg_all.mean():.3f}, std={norm_avg_all.std():.3f}"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_multiple_electrodes_raw(tb, raw, electrode_indices, allmat, monkey_name, trial_idx, output_path):
    """Plot 5: Raw data timeseries for 10 electrodes to show variety."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, electrode_idx in enumerate(electrode_indices):
        ax = axes[i]
        region = get_brain_region(electrode_idx, monkey_name)
        day = int(allmat[5, trial_idx])

        raw_trace = raw[:, electrode_idx, trial_idx]

        ax.plot(tb, raw_trace, linewidth=1.5, color=colors[i], alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Raw MUA', fontsize=10)
        ax.set_title(f'Electrode {electrode_idx} ({region}, Day {day})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add stats
        stats_text = f"mean={raw_trace.mean():.2f}, std={raw_trace.std():.2f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle(f'{monkey_name} - Raw Data for 10 Electrodes (Trial {trial_idx})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize normalization with 5 separate plots (all datapoints, no sampling)')
    parser.add_argument('--monkey', required=True, choices=['monkeyF', 'monkeyN'])
    parser.add_argument('--data_dir', default='/scratch/groups/anishm/tvsd/')
    parser.add_argument('--results_dir', default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/')
    parser.add_argument('--output_dir', default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/')
    parser.add_argument('--trial1_idx', type=int, default=None, help='First trial (default: auto-select)')
    parser.add_argument('--trial2_idx', type=int, default=None, help='Second trial (default: auto-select)')
    parser.add_argument('--electrode1_idx', type=int, default=None, help='First electrode (default: auto-select)')
    parser.add_argument('--electrode2_idx', type=int, default=None, help='Second electrode (default: auto-select)')
    parser.add_argument('--bin_width', type=int, default=10, help='Bin width in ms')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("NORMALIZATION COMPARISON (5 PLOTS)")
    print("="*60)

    # Load data once (efficient!)
    raw, normalized, allmat, tb, tb_norm = load_data(args.data_dir, args.results_dir, args.monkey)
    _, n_electrodes, n_trials = raw.shape

    # Select trials intelligently or use provided
    if args.trial1_idx is None or args.trial2_idx is None:
        trial1_idx, trial2_idx, electrode1_idx, electrode2_idx = select_diverse_trials(
            allmat, n_electrodes, args.monkey, args.seed
        )
    else:
        trial1_idx = np.clip(args.trial1_idx, 0, n_trials - 1)
        trial2_idx = np.clip(args.trial2_idx, 0, n_trials - 1)
        electrode1_idx = args.electrode1_idx if args.electrode1_idx is not None else np.random.randint(0, n_electrodes)
        electrode2_idx = args.electrode2_idx if args.electrode2_idx is not None else np.random.randint(0, n_electrodes)
        print(f"\nUsing Trial 1: trial={trial1_idx}, electrode={electrode1_idx}")
        print(f"Using Trial 2: trial={trial2_idx}, electrode={electrode2_idx}")

    # Extract data once (efficient reuse!)
    region1 = get_brain_region(electrode1_idx, args.monkey)
    region2 = get_brain_region(electrode2_idx, args.monkey)
    day1 = int(allmat[5, trial1_idx])
    day2 = int(allmat[5, trial2_idx])

    raw1 = raw[:, electrode1_idx, trial1_idx]
    raw2 = raw[:, electrode2_idx, trial2_idx]

    # Use ALL timepoints (no binning) - show every single datapoint
    print(f"\nUsing all normalized timepoints (no binning):")
    print(f"  Normalized timepoints: {normalized.shape[0]}")

    norm1 = normalized[:, electrode1_idx, trial1_idx]
    norm2 = normalized[:, electrode2_idx, trial2_idx]

    # Compute average across ALL trials for electrode1
    norm_avg_all = normalized[:, electrode1_idx, :].mean(axis=1)

    # Use the full time base (every single timepoint)
    time_bins = tb_norm

    print(f"\nTime axis information:")
    print(f"  Time range: {time_bins.min():.1f} to {time_bins.max():.1f} ms")
    print(f"  Pre-stimulus timepoints (t < 0): {(time_bins < 0).sum()}/{len(time_bins)}")
    print(f"  Post-stimulus timepoints (t >= 0): {(time_bins >= 0).sum()}/{len(time_bins)}")

    # Select 10 random electrodes from different regions for raw data visualization
    if args.monkey == 'monkeyF':
        regions = {'V1': range(0, 512), 'IT': range(512, 832), 'V4': range(832, 1024)}
    else:
        regions = {'V1': range(0, 512), 'V4': range(512, 768), 'IT': range(768, 1024)}

    # Sample ~3-4 electrodes from each region to get 10 total
    np.random.seed(args.seed + 1)
    electrodes_for_raw_vis = []
    n_per_region = [4, 3, 3]  # Total = 10
    for (region_name, region_range), n_samples in zip(regions.items(), n_per_region):
        selected = np.random.choice(list(region_range), size=n_samples, replace=False)
        electrodes_for_raw_vis.extend(selected)

    print(f"\nSelected 10 electrodes for raw data visualization: {electrodes_for_raw_vis}")

    print("\nGenerating plots...")

    # Create all 5 plots
    plot_single_raw_vs_norm(tb, time_bins, raw1, norm1, region1, day1, electrode1_idx,
                            args.monkey, os.path.join(args.output_dir, f'{args.monkey}_plot1_single_raw_vs_norm.png'))

    plot_two_trials_norm(time_bins, norm1, norm2, region1, region2, day1, day2,
                         args.monkey, os.path.join(args.output_dir, f'{args.monkey}_plot2_two_trials_norm.png'))

    plot_all_four(tb, time_bins, raw1, raw2, norm1, norm2, region1, region2,
                  args.monkey, os.path.join(args.output_dir, f'{args.monkey}_plot3_all_four.png'))

    plot_single_vs_average(time_bins, norm1, norm_avg_all, region1, day1, trial1_idx, electrode1_idx,
                           args.monkey, n_trials, os.path.join(args.output_dir, f'{args.monkey}_plot4_single_vs_average.png'))

    plot_multiple_electrodes_raw(tb, raw, electrodes_for_raw_vis, allmat, args.monkey, trial1_idx,
                                  os.path.join(args.output_dir, f'{args.monkey}_plot5_multiple_electrodes_raw.png'))

    print("\n" + "="*60)
    print("COMPLETE - 5 plots created")
    print("="*60)


if __name__ == '__main__':
    main()
