"""
Plot Trial-Averaged Raw MUA Data

Creates visualizations of trial-averaged raw MUA responses:
1. Trial-averaged timeseries for multiple electrodes
2. Trial-averaged heatmap across all electrodes, sorted by peak time

Usage:
    python plot_trial_averaged_raw.py --monkey monkeyF
    python plot_trial_averaged_raw.py --monkey monkeyN --n_electrodes 15
"""

import argparse
import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os


def load_raw_data(data_dir, monkey_name):
    """Load raw MUA data with channel mapping."""
    print(f"Loading raw data for {monkey_name}...")

    # Load raw data
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')
    with h5py.File(data_file, 'r') as f:
        allmua = np.array(f['ALLMUA'])  # (timepoints, trials, electrodes)
        allmat = np.array(f['ALLMAT'])  # (6, trials)
        tb = np.array(f['tb']).flatten()  # Time base in ms

    # Apply channel mapping
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    if not mapping_files:
        raise FileNotFoundError(f"No mapping file found for {monkey_name}")

    mapping_data = loadmat(os.path.join(data_dir, mapping_files[0]))
    mapping = mapping_data['mapping'].flatten() - 1  # Convert to 0-indexed
    allmua = allmua[..., mapping]

    print(f"  Raw shape: {allmua.shape} (timepoints, trials, electrodes)")
    print(f"  Time base: {tb.min():.1f} to {tb.max():.1f} ms ({len(tb)} timepoints)")
    print(f"  Number of trials: {allmua.shape[1]}")
    print(f"  Number of electrodes: {allmua.shape[2]}")

    return allmua, allmat, tb


def get_brain_region(electrode_idx, monkey_name):
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


def compute_trial_averaged(allmua):
    """
    Compute trial-averaged response for each electrode.

    Args:
        allmua: (n_timepoints, n_trials, n_electrodes)

    Returns:
        trial_avg: (n_timepoints, n_electrodes) - average across trials
    """
    print("\nComputing trial-averaged responses...")
    trial_avg = allmua.mean(axis=1)  # Average across trials
    print(f"  Trial-averaged shape: {trial_avg.shape}")
    return trial_avg


def find_peak_time(trial_avg, tb, baseline_end_idx=None):
    """
    Find the peak time for each electrode.

    Args:
        trial_avg: (n_timepoints, n_electrodes)
        tb: (n_timepoints,) - time base in ms
        baseline_end_idx: Only search for peaks after this index (to avoid baseline)

    Returns:
        peak_times: (n_electrodes,) - time of peak response in ms
        peak_indices: (n_electrodes,) - index of peak response
    """
    n_timepoints, n_electrodes = trial_avg.shape

    # Default: search after stimulus onset (t >= 0)
    if baseline_end_idx is None:
        baseline_end_idx = np.where(tb >= 0)[0][0]

    peak_indices = np.zeros(n_electrodes, dtype=int)
    peak_times = np.zeros(n_electrodes)

    for i in range(n_electrodes):
        # Find peak in post-stimulus period
        post_stim_response = trial_avg[baseline_end_idx:, i]
        peak_idx_relative = np.argmax(post_stim_response)
        peak_idx_absolute = baseline_end_idx + peak_idx_relative

        peak_indices[i] = peak_idx_absolute
        peak_times[i] = tb[peak_idx_absolute]

    return peak_times, peak_indices


def select_electrodes(n_electrodes_total, monkey_name, n_to_select=12, seed=42):
    """Select diverse electrodes from different regions."""
    np.random.seed(seed)

    if monkey_name == 'monkeyF':
        regions = {'V1': range(0, 512), 'IT': range(512, 832), 'V4': range(832, 1024)}
    else:
        regions = {'V1': range(0, 512), 'V4': range(512, 768), 'IT': range(768, 1024)}

    # Sample equally from each region
    n_per_region = n_to_select // 3
    remainder = n_to_select % 3

    selected = []
    for i, (region_name, region_range) in enumerate(regions.items()):
        n_samples = n_per_region + (1 if i < remainder else 0)
        region_electrodes = np.random.choice(list(region_range), size=n_samples, replace=False)
        selected.extend(region_electrodes)

    return sorted(selected)


def plot_trial_averaged_timeseries(tb, trial_avg, electrode_indices, monkey_name, output_path):
    """
    Plot trial-averaged timeseries for multiple electrodes.

    Args:
        tb: (n_timepoints,) - time base
        trial_avg: (n_timepoints, n_electrodes)
        electrode_indices: list of electrode indices to plot
        monkey_name: str
        output_path: str
    """
    n_electrodes = len(electrode_indices)
    n_cols = 3
    n_rows = int(np.ceil(n_electrodes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_electrodes > 1 else [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, n_electrodes))

    for i, electrode_idx in enumerate(electrode_indices):
        ax = axes[i]
        region = get_brain_region(electrode_idx, monkey_name)

        response = trial_avg[:, electrode_idx]

        ax.plot(tb, response, linewidth=2, color=colors[i], alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Stimulus onset')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Trial-Averaged MUA', fontsize=10)
        ax.set_title(f'Electrode {electrode_idx} ({region})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add peak time marker
        post_stim_idx = np.where(tb >= 0)[0][0]
        peak_idx = post_stim_idx + np.argmax(response[post_stim_idx:])
        peak_time = tb[peak_idx]
        peak_value = response[peak_idx]
        ax.plot(peak_time, peak_value, 'r*', markersize=15, label=f'Peak @ {peak_time:.1f}ms')

        ax.legend(loc='upper right', fontsize=8)

        # Add statistics
        stats_text = f"mean={response.mean():.2f}\nstd={response.std():.2f}\npeak={peak_value:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Hide empty subplots
    for i in range(n_electrodes, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{monkey_name} - Trial-Averaged Raw MUA Timeseries',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_trial_by_trial_heatmaps(tb, allmua, electrode_indices, monkey_name, output_path):
    """
    Plot trial-by-trial heatmaps for selected electrodes.
    Shows how individual trials vary around the mean response.

    Args:
        tb: (n_timepoints,) - time base
        allmua: (n_timepoints, n_trials, n_electrodes) - raw data
        electrode_indices: list of electrode indices to plot
        monkey_name: str
        output_path: str
    """
    print("\nGenerating trial-by-trial heatmaps...")

    n_electrodes = len(electrode_indices)
    n_cols = 3
    n_rows = int(np.ceil(n_electrodes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_electrodes > 1 else [axes]

    for i, electrode_idx in enumerate(electrode_indices):
        ax = axes[i]
        region = get_brain_region(electrode_idx, monkey_name)

        # Get all trials for this electrode: (n_timepoints, n_trials)
        electrode_data = allmua[:, :, electrode_idx]

        # Compute trial average for overlay
        trial_avg = electrode_data.mean(axis=1)

        # Transpose for heatmap: trials (rows) × time (columns)
        heatmap_data = electrode_data.T  # Now: (n_trials, n_timepoints)

        # Auto-scale colorbar based on data
        vmin = np.percentile(heatmap_data, 1)
        vmax = np.percentile(heatmap_data, 99)

        # Plot heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='hot',
                      extent=[tb[0], tb[-1], heatmap_data.shape[0], 0],
                      vmin=vmin, vmax=vmax, interpolation='nearest')

        # Overlay the trial average as a line
        # Need to normalize it to trial-space coordinates for visualization
        trial_avg_normalized = (trial_avg - vmin) / (vmax - vmin) * heatmap_data.shape[0]
        ax.plot(tb, trial_avg_normalized, color='cyan', linewidth=2.5,
               label='Trial average', linestyle='-', alpha=0.9)

        ax.axvline(0, color='lime', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus onset')

        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Trial Number', fontsize=10)
        ax.set_title(f'Electrode {electrode_idx} ({region})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        # Add colorbar
        divider = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        divider.set_label('MUA', fontsize=9)

        # Add statistics
        stats_text = (f"Trials: {heatmap_data.shape[0]}\n"
                     f"Mean: {trial_avg.mean():.2f}\n"
                     f"Std: {trial_avg.std():.2f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Hide empty subplots
    for i in range(n_electrodes, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{monkey_name} - Trial-by-Trial Response Heatmaps\n'
                 f'(Each row = one trial, color = MUA amplitude)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_region_comparison(tb, trial_avg, monkey_name, output_path):
    """
    Plot trial-averaged responses by brain region.

    Args:
        tb: (n_timepoints,) - time base
        trial_avg: (n_timepoints, n_electrodes)
        monkey_name: str
        output_path: str
    """
    print("\nGenerating region comparison plot...")

    if monkey_name == 'monkeyF':
        regions = {
            'V1': range(0, 512),
            'IT': range(512, 832),
            'V4': range(832, 1024)
        }
    else:
        regions = {
            'V1': range(0, 512),
            'V4': range(512, 768),
            'IT': range(768, 1024)
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'V1': '#E63946', 'V4': '#2A9D8F', 'IT': '#264653'}

    for ax, (region_name, region_range) in zip(axes, regions.items()):
        region_indices = list(region_range)
        region_responses = trial_avg[:, region_indices]

        # Plot mean and std across electrodes in this region
        mean_response = region_responses.mean(axis=1)
        std_response = region_responses.std(axis=1)

        ax.plot(tb, mean_response, linewidth=2.5, color=colors[region_name],
                label=f'{region_name} mean')
        ax.fill_between(tb, mean_response - std_response, mean_response + std_response,
                        alpha=0.3, color=colors[region_name], label='±1 std')

        # Also plot a few individual electrodes
        n_examples = 3
        example_indices = np.linspace(0, len(region_indices) - 1, n_examples, dtype=int)
        for i, idx in enumerate(example_indices):
            ax.plot(tb, region_responses[:, idx], linewidth=0.8, alpha=0.4,
                   color='gray', linestyle='--')

        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Trial-Averaged MUA', fontsize=11)
        ax.set_title(f'{region_name} (n={len(region_indices)} electrodes)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add peak time
        peak_times, _ = find_peak_time(region_responses, tb)
        median_peak = np.median(peak_times)
        ax.text(0.02, 0.98, f'Median peak: {median_peak:.1f}ms',
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle(f'{monkey_name} - Trial-Averaged Response by Brain Region',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot trial-averaged raw MUA data')
    parser.add_argument('--monkey', required=True, choices=['monkeyF', 'monkeyN'])
    parser.add_argument('--data_dir', default='/scratch/groups/anishm/tvsd/')
    parser.add_argument('--output_dir', default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/')
    parser.add_argument('--n_electrodes', type=int, default=12,
                       help='Number of electrodes to plot in timeseries and heatmap views')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("TRIAL-AVERAGED RAW MUA VISUALIZATION")
    print("="*60)

    # Load data
    allmua, allmat, tb = load_raw_data(args.data_dir, args.monkey)

    # Compute trial-averaged responses
    trial_avg = compute_trial_averaged(allmua)

    # Select electrodes for timeseries plots
    n_electrodes_total = trial_avg.shape[1]
    electrode_indices = select_electrodes(n_electrodes_total, args.monkey,
                                         n_to_select=args.n_electrodes, seed=args.seed)
    print(f"\nSelected {len(electrode_indices)} electrodes for timeseries plots:")
    for idx in electrode_indices:
        region = get_brain_region(idx, args.monkey)
        print(f"  Electrode {idx} ({region})")

    # Generate plots
    print("\nGenerating plots...")

    # 1. Trial-averaged timeseries for selected electrodes
    plot_trial_averaged_timeseries(
        tb, trial_avg, electrode_indices, args.monkey,
        os.path.join(args.output_dir, f'{args.monkey}_trial_avg_timeseries.png')
    )

    # 2. Trial-by-trial heatmaps (shows variability across trials)
    plot_trial_by_trial_heatmaps(
        tb, allmua, electrode_indices, args.monkey,
        os.path.join(args.output_dir, f'{args.monkey}_trial_by_trial_heatmaps.png')
    )

    # 3. Region comparison
    plot_region_comparison(
        tb, trial_avg, args.monkey,
        os.path.join(args.output_dir, f'{args.monkey}_trial_avg_by_region.png')
    )

    print("\n" + "="*60)
    print("COMPLETE - 3 plots created:")
    print(f"  1. {args.monkey}_trial_avg_timeseries.png - Trial-averaged timeseries")
    print(f"  2. {args.monkey}_trial_by_trial_heatmaps.png - Trial-by-trial variability")
    print(f"  3. {args.monkey}_trial_avg_by_region.png - Regional comparison")
    print("="*60)


if __name__ == '__main__':
    main()
