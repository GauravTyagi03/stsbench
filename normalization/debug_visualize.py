"""
Debug Normalization with Diagnostic Visualizations

This script generates diagnostic plots to understand the normalization pipeline:
1. Baseline vs Raw - Verify Stage 1 works correctly
2. Final vs Baseline - Isolate Stage 2 issues
3. Neural Heatmaps - Show activation patterns across all electrodes

Usage:
    python normalization/debug_visualize.py \
        --monkey monkeyF \
        --data_dir /scratch/groups/anishm/tvsd/ \
        --results_dir /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/ \
        --output_dir normalization/debug_plots/
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_channel_mapping(data_dir, monkey_name):
    """Load channel mapping for electrode reordering."""
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    if not mapping_files:
        raise FileNotFoundError(f"Could not find mapping file for {monkey_name} in {data_dir}")

    mapping_file = os.path.join(data_dir, mapping_files[0])
    print(f"Using mapping file: {mapping_files[0]}")

    mapping_data = loadmat(mapping_file)
    mapping = mapping_data['mapping'].flatten() - 1  # Convert to 0-indexed
    return mapping


def load_all_normalization_stages(data_dir, results_dir, monkey_name):
    """Load raw, baseline-normalized, and final-normalized data."""
    print("\nLoading all normalization stages...")

    # Load raw data
    raw_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')
    print(f"Loading raw data from {raw_file}")
    with h5py.File(raw_file, 'r') as f:
        raw = np.array(f['ALLMUA'])
        allmat = np.array(f['ALLMAT'])
        tb = np.array(f['tb']).flatten()

    # Apply channel mapping
    mapping = load_channel_mapping(data_dir, monkey_name)
    raw = raw[..., mapping].transpose(0, 2, 1)

    # Load baseline-normalized
    baseline_file = os.path.join(results_dir, f'{monkey_name}_baseline_normalized.h5')
    print(f"Loading baseline-normalized data from {baseline_file}")
    with h5py.File(baseline_file, 'r') as f:
        baseline_norm = f['baseline_normalized'][:]
        tb_baseline = f['tb'][:]

    # Load final-normalized
    final_file = os.path.join(results_dir, f'{monkey_name}_timeseries_normalized.h5')
    print(f"Loading final-normalized data from {final_file}")
    with h5py.File(final_file, 'r') as f:
        final_norm = f['timeseries_normalized'][:]
        tb_final = f['tb'][:]

    return raw, baseline_norm, final_norm, allmat, tb, tb_baseline, tb_final


def plot_baseline_vs_raw(raw, baseline_norm, tb, trial_idx, electrode_idx, output_path):
    """Compare baseline-normalized output with raw data for a single trial."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot raw
    axes[0].plot(tb, raw[:, electrode_idx, trial_idx], 'b-', linewidth=1.5)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus onset')
    axes[0].set_ylabel('Raw MUA', fontsize=11)
    axes[0].set_title(f'Raw Data (Trial {trial_idx}, Electrode {electrode_idx})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot baseline-normalized
    axes[1].plot(tb, baseline_norm[:, electrode_idx, trial_idx], 'g-', linewidth=1.5)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus onset')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Baseline-Normalized MUA', fontsize=11)
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_title(f'After Stage 1: Baseline Normalization', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Add statistics
    stats_text = (
        f"Raw: mean={raw[:, electrode_idx, trial_idx].mean():.2f}, "
        f"std={raw[:, electrode_idx, trial_idx].std():.2f}\n"
        f"Baseline-norm: mean={baseline_norm[:, electrode_idx, trial_idx].mean():.4f}, "
        f"std={baseline_norm[:, electrode_idx, trial_idx].std():.4f}"
    )
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_final_vs_baseline(baseline_norm, final_norm, tb_baseline, tb_final,
                           trial_idx, electrode_idx, day, bin_width, output_path):
    """Compare final normalized with baseline normalized to isolate Stage 2 effect."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot baseline-normalized
    axes[0].plot(tb_baseline, baseline_norm[:, electrode_idx, trial_idx], 'g-', linewidth=1.5)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus onset')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Baseline-Normalized', fontsize=11)
    axes[0].set_title(f'After Stage 1 (Trial {trial_idx}, Electrode {electrode_idx}, Day {day})',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot final-normalized
    axes[1].plot(tb_final, final_norm[:, electrode_idx, trial_idx], 'orange', linewidth=2,
                marker='o', markersize=3, label='Final normalized')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus onset')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Final Normalized', fontsize=11)
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_title(f'After Stage 2: Bin Normalization (bin_width={bin_width}ms)',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Add statistics
    baseline_data = baseline_norm[:len(tb_final), electrode_idx, trial_idx]
    final_data = final_norm[:, electrode_idx, trial_idx]

    stats_text = (
        f"Baseline-norm: mean={baseline_data.mean():.4f}, std={baseline_data.std():.4f}\n"
        f"Final-norm: mean={final_data.mean():.4f}, std={final_data.std():.4f}\n"
        f"Non-zero fraction: {(np.abs(final_data) > 0.001).sum() / len(final_data):.2%}"
    )
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_neural_heatmap(data, tb, trial_idx, monkey_name, stage_name, output_path):
    """
    Create heatmap showing average activation across all electrodes for a specific trial.

    Args:
        data: (n_timepoints, n_electrodes, n_trials)
        tb: time base array
        trial_idx: which trial to visualize
        stage_name: 'raw', 'baseline_norm', or 'final_norm'
    """
    # Extract data for this trial: (n_timepoints, n_electrodes)
    trial_data = data[:, :, trial_idx]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Create heatmap (electrodes on y-axis, time on x-axis)
    im = ax.imshow(trial_data.T, aspect='auto', cmap='RdBu_r',
                   extent=[tb.min(), tb.max(), 0, trial_data.shape[1]],
                   interpolation='nearest', vmin=-3, vmax=3)

    # Add stimulus onset line
    ax.axvline(0, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label='Stimulus onset')

    # Labels and title
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Electrode Index', fontsize=12)
    ax.set_title(f'{monkey_name} - Neural Activation Heatmap (Trial {trial_idx}, {stage_name})\n'
                f'All {trial_data.shape[1]} Electrodes × Time',
                fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation (normalized)', fontsize=11)

    # Add brain region markers (for monkeyF)
    if monkey_name == 'monkeyF':
        ax.axhline(512, color='white', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axhline(832, color='white', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.text(tb.min() + 10, 256, 'V1', fontsize=10, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        ax.text(tb.min() + 10, 672, 'IT', fontsize=10, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        ax.text(tb.min() + 10, 928, 'V4', fontsize=10, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    # Statistics annotation
    stats_text = (
        f"Mean: {trial_data.mean():.4f}\n"
        f"Std: {trial_data.std():.4f}\n"
        f"Min: {trial_data.min():.4f}\n"
        f"Max: {trial_data.max():.4f}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Debug normalization with diagnostic plots')
    parser.add_argument('--monkey', required=True, choices=['monkeyF', 'monkeyN'])
    parser.add_argument('--data_dir', default='/scratch/groups/anishm/tvsd/')
    parser.add_argument('--results_dir', default='/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/')
    parser.add_argument('--output_dir', default='normalization/debug_plots/')
    parser.add_argument('--trial_idx', type=int, default=None, help='Trial to visualize (default: random)')
    parser.add_argument('--electrode_idx', type=int, default=None, help='Electrode to visualize (default: random)')
    parser.add_argument('--bin_width', type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("NORMALIZATION DEBUGGING VISUALIZATION")
    print("="*60)

    # Load all stages
    raw, baseline_norm, final_norm, allmat, tb, tb_baseline, tb_final = \
        load_all_normalization_stages(args.data_dir, args.results_dir, args.monkey)

    print(f"\nLoaded data shapes:")
    print(f"  Raw: {raw.shape}")
    print(f"  Baseline-normalized: {baseline_norm.shape}")
    print(f"  Final-normalized: {final_norm.shape}")

    # Select trial and electrode
    trial_idx = args.trial_idx if args.trial_idx is not None else np.random.randint(0, raw.shape[2])
    electrode_idx = args.electrode_idx if args.electrode_idx is not None else np.random.randint(0, raw.shape[1])
    day = int(allmat[5, trial_idx])

    print(f"\nSelected trial={trial_idx}, electrode={electrode_idx}, day={day}")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")

    # Plot 1: Baseline vs Raw
    plot_baseline_vs_raw(
        raw, baseline_norm, tb,
        trial_idx, electrode_idx,
        os.path.join(args.output_dir, f'{args.monkey}_debug_baseline_vs_raw.png')
    )

    # Plot 2: Final vs Baseline
    plot_final_vs_baseline(
        baseline_norm, final_norm, tb_baseline, tb_final,
        trial_idx, electrode_idx, day, args.bin_width,
        os.path.join(args.output_dir, f'{args.monkey}_debug_final_vs_baseline.png')
    )

    # Plot 3: Neural heatmap for raw data
    plot_neural_heatmap(
        raw, tb, trial_idx, args.monkey, 'Raw',
        os.path.join(args.output_dir, f'{args.monkey}_heatmap_raw.png')
    )

    # Plot 4: Neural heatmap for baseline-normalized
    plot_neural_heatmap(
        baseline_norm, tb_baseline, trial_idx, args.monkey, 'Baseline Normalized',
        os.path.join(args.output_dir, f'{args.monkey}_heatmap_baseline_norm.png')
    )

    # Plot 5: Neural heatmap for final-normalized
    plot_neural_heatmap(
        final_norm, tb_final, trial_idx, args.monkey, 'Final Normalized',
        os.path.join(args.output_dir, f'{args.monkey}_heatmap_final_norm.png')
    )

    print("\n" + "="*60)
    print("COMPLETE - Diagnostic plots created")
    print("="*60)


if __name__ == '__main__':
    main()
