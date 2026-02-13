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


def plot_heatmap_sorted_by_peak(tb, trial_avg, monkey_name, output_path, vmin=None, vmax=None):
    """
    Plot heatmap of trial-averaged responses sorted by peak time.

    Args:
        tb: (n_timepoints,) - time base
        trial_avg: (n_timepoints, n_electrodes)
        monkey_name: str
        output_path: str
        vmin, vmax: colorbar limits (optional)
    """
    print("\nGenerating heatmap sorted by peak time...")

    # Find peak times
    peak_times, peak_indices = find_peak_time(trial_avg, tb)

    # Sort electrodes by peak time
    sort_order = np.argsort(peak_times)
    trial_avg_sorted = trial_avg[:, sort_order]
    peak_times_sorted = peak_times[sort_order]

    print(f"  Peak time range: {peak_times.min():.1f} to {peak_times.max():.1f} ms")
    print(f"  Median peak time: {np.median(peak_times):.1f} ms")

    # Create figure with two subplots: heatmap and peak time distribution
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[20, 1],
                          hspace=0.3, wspace=0.05)

    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[0, 0])

    # Auto-scale colorbar if not provided
    if vmin is None:
        vmin = np.percentile(trial_avg_sorted, 1)
    if vmax is None:
        vmax = np.percentile(trial_avg_sorted, 99)

    im = ax_heatmap.imshow(trial_avg_sorted.T, aspect='auto', cmap='RdBu_r',
                           extent=[tb[0], tb[-1], trial_avg_sorted.shape[1], 0],
                           vmin=vmin, vmax=vmax, interpolation='nearest')

    ax_heatmap.axvline(0, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label='Stimulus onset')
    ax_heatmap.set_xlabel('Time (ms)', fontsize=12)
    ax_heatmap.set_ylabel('Electrode (sorted by peak time)', fontsize=12)
    ax_heatmap.set_title(f'{monkey_name} - Trial-Averaged Raw MUA Heatmap (Sorted by Peak Time)',
                         fontsize=14, fontweight='bold')
    ax_heatmap.legend(loc='upper right', fontsize=10)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Trial-Averaged MUA', fontsize=11)

    # Peak time distribution
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.hist(peak_times, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_dist.axvline(np.median(peak_times), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(peak_times):.1f}ms')
    ax_dist.set_xlabel('Peak Time (ms)', fontsize=12)
    ax_dist.set_ylabel('Number of Electrodes', fontsize=12)
    ax_dist.set_title('Distribution of Peak Times', fontsize=12, fontweight='bold')
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)

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
                       help='Number of electrodes to plot in timeseries view')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vmin', type=float, default=None,
                       help='Colorbar min for heatmap (default: auto)')
    parser.add_argument('--vmax', type=float, default=None,
                       help='Colorbar max for heatmap (default: auto)')

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

    # 2. Heatmap sorted by peak time
    plot_heatmap_sorted_by_peak(
        tb, trial_avg, args.monkey,
        os.path.join(args.output_dir, f'{args.monkey}_trial_avg_heatmap.png'),
        vmin=args.vmin, vmax=args.vmax
    )

    # 3. Region comparison
    plot_region_comparison(
        tb, trial_avg, args.monkey,
        os.path.join(args.output_dir, f'{args.monkey}_trial_avg_by_region.png')
    )

    print("\n" + "="*60)
    print("COMPLETE - 3 plots created:")
    print(f"  1. {args.monkey}_trial_avg_timeseries.png - Individual electrode timeseries")
    print(f"  2. {args.monkey}_trial_avg_heatmap.png - Heatmap sorted by peak time")
    print(f"  3. {args.monkey}_trial_avg_by_region.png - Regional comparison")
    print("="*60)


if __name__ == '__main__':
    main()
