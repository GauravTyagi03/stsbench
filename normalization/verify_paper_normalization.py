"""
Verify Paper's Normalization Method

Implements and validates the paper's normalization:
1. Average MUA response across region-specific time windows
2. Normalize per (electrode, day) using TEST pool statistics only
3. Average by stimulus ID

Usage:
    python verify_paper_normalization.py --monkey monkeyF
"""

import argparse
import h5py
import numpy as np
import scipy.io
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import pdist
import os
import matplotlib.pyplot as plt


# Brain regions and time windows by monkey
BRAIN_REGIONS = {
    'monkeyF': {
        'V1': {'electrodes': range(0, 512), 'time_window': (25, 125)},
        'IT': {'electrodes': range(512, 832), 'time_window': (75, 175)},
        'V4': {'electrodes': range(832, 1024), 'time_window': (50, 150)}
    },
    'monkeyN': {
        'V1': {'electrodes': range(0, 512), 'time_window': (25, 125)},
        'V4': {'electrodes': range(512, 768), 'time_window': (50, 150)},
        'IT': {'electrodes': range(768, 1024), 'time_window': (75, 175)}
    }
}


def load_h5_mat(filepath):
    """Load MATLAB v7.3 (HDF5) file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if not key.startswith('#'):
                data[key] = np.array(f[key])
    return data


def load_mua_data(data_dir, monkey_name):
    """Load MUA data with channel mapping."""
    print(f"Loading {monkey_name} data...")

    # Load main data
    data_file = os.path.join(data_dir, f'{monkey_name}_THINGS_MUA_trials.mat')
    with h5py.File(data_file, 'r') as f:
        allmua = np.array(f['ALLMUA'])  # (timepoints, trials, electrodes)
        allmat = np.array(f['ALLMAT'])  # (6, trials)
        tb = np.array(f['tb']).flatten()  # Time base vector (in ms)

    # Load channel mapping
    mapping_files = [f for f in os.listdir(data_dir)
                     if f.startswith(f"{monkey_name}_1024chns_mapping") and f.endswith('.mat')]
    mapping_data = scipy.io.loadmat(os.path.join(data_dir, mapping_files[0]))
    mapping = mapping_data['mapping'].flatten() - 1  # Convert to 0-indexed

    allmua = allmua[..., mapping]
    print(f"Loaded shape: {allmua.shape} (timepoints, trials, electrodes)")
    print(f"Time base: {tb.min():.1f} to {tb.max():.1f} ms, {len(tb)} points")

    # Verify tb matches allmua shape
    assert len(tb) == allmua.shape[0], f"tb length {len(tb)} != allmua timepoints {allmua.shape[0]}"

    return allmua, allmat, tb


def normalize_mua(allmua, allmat, brain_regions, tb):
    """
    Normalize MUA using test pool statistics per (electrode, day).

    Args:
        allmua: (n_timepoints, n_trials, n_electrodes)
        allmat: (6, n_trials) - metadata with [_, train_idx, test_idx, _, _, day]
        brain_regions: Dict of region configs
        tb: (n_timepoints,) - time base vector in ms

    Returns:
        normalized: (n_electrodes, n_trials)
    """
    n_electrodes = allmua.shape[2]
    n_trials = allmua.shape[1]

    # Extract metadata
    test_mask = allmat[2] > 0
    days = allmat[5].astype(int)

    normalized = np.zeros((n_electrodes, n_trials))

    # Get electrode -> region mapping with correct time indices
    electrode_to_indices = {}
    for region, config in brain_regions.items():
        start_ms, end_ms = config['time_window']
        # Match MATLAB logic: tb > start_ms & tb <= end_ms
        time_mask = (tb > start_ms) & (tb <= end_ms)
        time_indices = np.where(time_mask)[0]

        for elec_idx in config['electrodes']:
            electrode_to_indices[elec_idx] = time_indices

    print(f"Normalizing {n_electrodes} electrodes...")
    print(f"  Example time window V1 (25-125ms): indices {electrode_to_indices[0][:3]}...{electrode_to_indices[0][-3:]}")

    for elec_idx in range(n_electrodes):
        if elec_idx % 200 == 0:
            print(f"  Processing electrode {elec_idx}/{n_electrodes}")

        # Extract time-windowed data using correct indices
        time_indices = electrode_to_indices[elec_idx]
        windowed_data = allmua[time_indices, :, elec_idx]  # (n_timepoints_in_window, n_trials)

        # Normalize per day using test pool
        for day in np.unique(days):
            day_mask = days == day
            test_day_mask = day_mask & test_mask

            # Get 2D test data for this day (timepoints × trials)
            test_day_2d = windowed_data[:, test_day_mask]

            if test_day_2d.shape[1] == 0:
                normalized[elec_idx, day_mask] = 0
                continue

            # Compute mean and std across ALL values (trials AND timepoints)
            mean = test_day_2d.mean()
            std = test_day_2d.std()  # Use ddof=0 (default) to match paper's normalization
            std = 1.0 if std == 0 or np.isnan(std) else std

            # Average across time for normalization, then normalize
            day_data_2d = windowed_data[:, day_mask]
            elec_data_day = day_data_2d.mean(axis=0)  # Average time for this day
            normalized[elec_idx, day_mask] = (elec_data_day - mean) / std

    return normalized


def average_by_stimulus(normalized, allmat):
    """
    Average normalized data by stimulus ID.

    Returns:
        train_MUA: (n_train_stimuli, n_electrodes)
        test_MUA: (n_test_stimuli, n_electrodes)
        test_MUA_reps: (n_electrodes, n_test_stimuli, n_reps)
    """
    train_idx = allmat[1].astype(int)
    test_idx = allmat[2].astype(int)

    unique_train = np.unique(train_idx[train_idx > 0])
    unique_test = np.unique(test_idx[test_idx > 0])

    n_electrodes = normalized.shape[0]

    # Average train stimuli
    train_MUA = np.array([normalized[:, train_idx == stim_id].mean(axis=1)
                          for stim_id in unique_train])

    # Average test stimuli
    test_MUA = np.array([normalized[:, test_idx == stim_id].mean(axis=1)
                         for stim_id in unique_test])

    # Collect test repetitions
    max_reps = max(np.sum(test_idx == stim_id) for stim_id in unique_test)
    test_MUA_reps = np.zeros((n_electrodes, len(unique_test), max_reps))

    for i, stim_id in enumerate(unique_test):
        stim_trials = np.where(test_idx == stim_id)[0]
        test_MUA_reps[:, i, :len(stim_trials)] = normalized[:, stim_trials]

    print(f"Averaged: {len(unique_train)} train, {len(unique_test)} test stimuli")
    return train_MUA, test_MUA, test_MUA_reps


def compute_reliability_oracle(test_MUA_reps):
    """
    Compute reliability and oracle correlation from test repetitions.
    Matches MATLAB logic: correlations across stimuli between repetitions.

    Args:
        test_MUA_reps: (n_electrodes, n_test_stimuli, n_reps)

    Returns:
        reliab: (n_electrodes, n_pairs) - pairwise correlations between reps
        oracle: (n_electrodes,) - average oracle correlation
    """
    n_electrodes, n_stimuli, n_reps = test_MUA_reps.shape

    reliab = []
    oracle = np.zeros(n_electrodes)

    print("\nComputing reliability and oracle...")
    for elec in range(n_electrodes):
        if elec % 200 == 0:
            print(f"  Processing electrode {elec}/{n_electrodes}")

        # Get data for this electrode: (n_stimuli, n_reps)
        elec_data = test_MUA_reps[elec, :, :]

        # Reliability: pairwise correlations between reps (across stimuli)
        # pdist computes pairwise correlation distances between columns
        pairwise_corr = 1 - pdist(elec_data.T, metric='correlation')
        reliab.append(pairwise_corr)

        # Oracle: correlation between each rep and mean of others
        oracle_vals = []
        for rep in range(n_reps):
            rep_data = elec_data[:, rep]
            # Mean of all other reps
            other_reps_idx = [r for r in range(n_reps) if r != rep]
            if len(other_reps_idx) > 0:
                mean_others = np.mean(elec_data[:, other_reps_idx], axis=1)
                # Correlation between this rep and mean of others
                corr, _ = pearsonr(rep_data, mean_others)
                if not np.isnan(corr):
                    oracle_vals.append(corr)

        oracle[elec] = np.mean(oracle_vals) if oracle_vals else np.nan

    # Convert reliab to 2D array (pad shorter arrays with nan)
    max_pairs = max(len(r) for r in reliab)
    reliab_array = np.full((n_electrodes, max_pairs), np.nan)
    for i, r in enumerate(reliab):
        reliab_array[i, :len(r)] = r

    print(f"  Mean oracle: {np.nanmean(oracle):.4f}")
    return reliab_array, oracle


def validate_and_plot(our_train, our_test, orig_train, orig_test, brain_regions,
                      monkey_name, output_dir):
    """Validate normalization and create comparison plots."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    # Add "ALL" region
    regions = {**brain_regions, 'ALL': {'electrodes': range(our_train.shape[1])}}

    for dataset_name, our_data, orig_data in [('train', our_train, orig_train),
                                               ('test', our_test, orig_test)]:
        print(f"\n{dataset_name.upper()} Dataset:")

        # Create validation plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        region_list = list(regions.keys())
        our_means = []
        orig_means = []
        our_stds = []
        orig_stds = []

        for region_name, region_info in regions.items():
            elecs = list(region_info['electrodes'])
            our_region = our_data[:, elecs].flatten()
            orig_region = orig_data[:, elecs].flatten()

            our_mean, our_std = our_region.mean(), our_region.std(ddof=1)
            orig_mean, orig_std = orig_region.mean(), orig_region.std(ddof=1)

            our_means.append(our_mean)
            orig_means.append(orig_mean)
            our_stds.append(our_std)
            orig_stds.append(orig_std)

            # KS test
            ks_stat, ks_pval = ks_2samp(our_region, orig_region)

            print(f"  {region_name:4s}: Mean diff={abs(our_mean - orig_mean):.4f}, "
                  f"Std ratio={our_std/orig_std:.4f}, KS p={ks_pval:.4f}")

        # Plot mean comparison
        x = np.arange(len(region_list))
        width = 0.35
        axes[0, 0].bar(x - width/2, orig_means, width, label='Original', alpha=0.8)
        axes[0, 0].bar(x + width/2, our_means, width, label='Ours', alpha=0.8)
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].set_title('Mean by Region')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(region_list)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot std comparison
        axes[0, 1].bar(x - width/2, orig_stds, width, label='Original', alpha=0.8)
        axes[0, 1].bar(x + width/2, our_stds, width, label='Ours', alpha=0.8)
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].set_title('Std by Region')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(region_list)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Hide unused top panels
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')

        # Plot distributions for each region
        for idx, region_name in enumerate(['V1', 'V4', 'IT', 'ALL']):
            ax = axes[1, idx]
            elecs = list(regions[region_name]['electrodes'])
            our_region = our_data[:, elecs].flatten()
            orig_region = orig_data[:, elecs].flatten()

            bins = np.linspace(-10, 10, 100) if dataset_name == 'train' else 50
            ax.hist(orig_region, bins=bins, alpha=0.5, label='Original', density=True)
            ax.hist(our_region, bins=bins, alpha=0.5, label='Ours', density=True)
            ax.set_xlim(-3, 3)
            ax.set_xlabel('MUA Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{region_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{dataset_name.upper()} Dataset - Validation',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(output_dir, f'{monkey_name}_{dataset_name}_validation_plots.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Verify paper normalization method')
    parser.add_argument('--monkey', required=True, choices=['monkeyF', 'monkeyN'])
    parser.add_argument('--data_dir', default='/scratch/groups/anishm/tvsd/')
    parser.add_argument('--output_dir', default='/oak/stanford/groups/anishm/gtyagi/stsbench/results/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    allmua, allmat, tb = load_mua_data(args.data_dir, args.monkey)

    # Normalize
    print("\nApplying normalization...")
    brain_regions = BRAIN_REGIONS[args.monkey]
    normalized = normalize_mua(allmua, allmat, brain_regions, tb)
    train_MUA, test_MUA, test_MUA_reps = average_by_stimulus(normalized, allmat)

    print(f"\nResults:")
    print(f"  train_MUA: {train_MUA.shape}")
    print(f"  test_MUA: {test_MUA.shape}")
    print(f"  test_MUA_reps: {test_MUA_reps.shape}")

    # Compute reliability and oracle
    reliab, oracle = compute_reliability_oracle(test_MUA_reps)

    # Save in HDF5/MATLAB v7.3 format for compatibility with h5py
    output_file = os.path.join(args.output_dir, f'{args.monkey}_paper_normalized.mat')
    scipy.io.savemat(output_file, {
        'train_MUA': train_MUA,
        'test_MUA': test_MUA,
        'test_MUA_reps': test_MUA_reps,
        'reliab': reliab,
        'oracle': oracle
    }, do_compression=True)  # Creates HDF5/MATLAB v7.3 format
    print(f"\nSaved: {output_file} (HDF5 format)")

    # Validate if original exists
    try:
        original_file = os.path.join(args.data_dir, f'{args.monkey}_THINGS_normMUA.mat')
        original_data = load_h5_mat(original_file)

        validate_and_plot(
            train_MUA, test_MUA,
            original_data['train_MUA'], original_data['test_MUA'],
            brain_regions, args.monkey, args.output_dir
        )
    except FileNotFoundError:
        print(f"\nWarning: Original file not found, skipping validation")


if __name__ == '__main__':
    main()
