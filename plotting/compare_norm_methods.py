#!/usr/bin/env python3
"""
Compare different MUA normalization methods for the THINGS ventral-stream dataset.

This script implements and compares:
  1. Option A: Pure per-bin/per-electrode z-scoring (post-binning)
  2. Option B: 100 ms baseline normalization only
  3. Option C: 100 ms baseline normalization + per-bin/per-electrode z-scoring (RECOMMENDED)
  
Additionally, we attempt to reconstruct the original THINGSnormMUA to validate our approach.

Usage:
    python compare_norm_methods.py --data_dir /path/to/tvsd_data --monkey monkeyF --output_dir ./results

Requirements:
    - scipy, numpy, h5py, scipy
    - monkeyF_THINGS_MUA_trials.mat
    - monkeyF_things_imgs.mat
    - monkeyF_1024chns_mapping_XXXXXXXX.mat
    - (optional) monkeyF_THINGS_normMUA.mat for validation
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import h5py
from scipy.io import loadmat, savemat
from scipy import stats
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ============================================================================
# Data Loading Utilities
# ============================================================================

def _ref_to_str(f, ref) -> str:
    """Convert HDF5 reference to string."""
    v = f[ref][()]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    v = np.array(v)
    if v.dtype.kind in {"S", "U"}:
        return "".join(v.astype(str).flatten())
    if v.dtype.kind in {"i", "u"}:
        return "".join(map(chr, v.flatten()))
    return str(v)


def load_mua_trials(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw MUA trials from TVSD.
    
    Returns:
        allmua: (n_timepoints, n_electrodes, n_trials) - raw MUA
        tb: time bins / timepoint indices
        train_stim_indices: (n_train_trials,) stimulus indices for training
        test_stim_indices: (n_test_trials,) stimulus indices for test
    """
    data_filename = f"{monkey_name}_THINGS_MUA_trials.mat"
    file_path = os.path.join(data_dir, data_filename)
    
    print(f"Loading MUA trials from {file_path}...")
    with h5py.File(file_path, "r") as f:
        allmua = f["ALLMUA"][...]
        allmat = f["ALLMAT"][...]
        tb = f["tb"][...]
    
    # Load channel mapping and reorder
    mapping_files = [f for f in os.listdir(data_dir) if f"{monkey_name}_1024chns_mapping" in f]
    if not mapping_files:
        raise FileNotFoundError(f"Could not find mapping file for {monkey_name} in {data_dir}")
    
    mapping_file = os.path.join(data_dir, mapping_files[0])
    mapping = loadmat(mapping_file)["mapping"].flatten() - 1  # MATLAB to Python indexing
    allmua = allmua[..., mapping]
    
    train_stim_indices = allmat[1].astype(np.int32)
    test_stim_indices = allmat[2].astype(np.int32)
    
    print(f"  Shape: {allmua.shape} (timepoints, electrodes, trials)")
    print(f"  Training trials: {len(train_stim_indices)}")
    print(f"  Test trials: {len(test_stim_indices)}")
    print(f"  Timepoints per trial: {len(tb)}")
    
    return allmua, tb, train_stim_indices, test_stim_indices


def load_norm_mua_original(data_dir: str, monkey_name: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempt to load the original THINGSnormMUA from authors.
    
    Returns:
        dict with 'norm_mua_train', 'norm_mua_test', 'reliability' if found, else None
    """
    filename = f"{monkey_name}_THINGS_normMUA.mat"
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"Note: Original {filename} not found. Skipping validation against original normalization.")
        return None
    
    print(f"Loading original normalized MUA from {file_path}...")
    data = loadmat(file_path)
    
    result = {}
    for key in data.keys():
        if not key.startswith("__"):
            result[key] = data[key]
            if result[key].ndim > 0:
                print(f"  {key}: {result[key].shape}")
    
    return result


# ============================================================================
# Preprocessing & Normalization Methods
# ============================================================================

class NormalizationPipeline:
    """Base class for normalization pipelines."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.stats = {}  # Store normalization statistics
    
    def fit(self, data: np.ndarray, train_idx: np.ndarray) -> None:
        """Fit normalization parameters on training data."""
        raise NotImplementedError
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply learned normalization to data."""
        raise NotImplementedError
    
    def fit_transform(self, data: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data, train_idx)
        return self.transform(data)


class OptionA(NormalizationPipeline):
    """
    Option A: Pure per-bin/per-electrode z-scoring (post-binning).
    
    After binning:
      - Compute mean and std per (bin, electrode) using training trials
      - z-score all splits using training stats
    """
    
    def __init__(self, bin_size_ms: int = 20):
        super().__init__(
            name="Option A: Per-Bin/Per-Electrode z-score",
            description="Pure z-scoring per (bin, electrode) without baseline subtraction"
        )
        self.bin_size_ms = bin_size_ms
        self.binned_data = None
        self.bin_means = None
        self.bin_stds = None
    
    def _bin_data(self, data: np.ndarray) -> np.ndarray:
        """Bin raw timepoint data into temporal bins."""
        n_timepoints, n_elec, n_trials = data.shape
        n_bins = n_timepoints // self.bin_size_ms
        
        binned = np.zeros((n_bins, n_elec, n_trials), dtype=np.float32)
        for b in range(n_bins):
            start_idx = b * self.bin_size_ms
            end_idx = start_idx + self.bin_size_ms
            binned[b] = data[start_idx:end_idx].mean(axis=0)
        
        return binned
    
    def fit(self, data: np.ndarray, train_idx: np.ndarray) -> None:
        """Compute mean and std per bin/electrode from training trials."""
        self.binned_data = self._bin_data(data)
        
        # Compute stats on training trials only
        train_binned = self.binned_data[:, :, train_idx]
        
        self.bin_means = train_binned.mean(axis=2, keepdims=True)
        self.bin_stds = train_binned.std(axis=2, keepdims=True)
        
        # Avoid division by zero
        self.bin_stds[self.bin_stds == 0] = 1.0
        
        self.stats = {
            "bin_means_shape": self.bin_means.shape,
            "bin_stds_shape": self.bin_stds.shape,
            "mean_of_means": float(self.bin_means.mean()),
            "mean_of_stds": float(self.bin_stds.mean()),
        }
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-scoring to binned data."""
        if self.binned_data is None or self.bin_means is None:
            raise RuntimeError("Must call fit() before transform()")
        
        normalized = (self.binned_data - self.bin_means) / self.bin_stds
        return normalized


class OptionB(NormalizationPipeline):
    """
    Option B: 100 ms baseline normalization only (no per-bin z-scoring).
    
    Before binning:
      - Use first 100 ms as baseline window
      - Subtract and divide by baseline mean/std per electrode per trial
      - Interpret result as "baseline-z" throughout the trial
    """
    
    def __init__(self, baseline_window_ms: int = 100, bin_size_ms: int = 20):
        super().__init__(
            name="Option B: Baseline Normalization Only",
            description="100ms baseline subtraction per electrode (no per-bin z-score)"
        )
        self.baseline_window_ms = baseline_window_ms
        self.bin_size_ms = bin_size_ms
        self.baseline_means = None
        self.baseline_stds = None
        self.baseline_normalized_data = None
    
    def fit(self, data: np.ndarray, train_idx: np.ndarray) -> None:
        """
        Compute baseline statistics from first N ms (all trials for baseline).
        Note: baseline is computed from all trials as it's stimulus-independent.
        """
        n_timepoints = data.shape[0]
        baseline_idx = slice(0, self.baseline_window_ms)
        
        baseline_data = data[baseline_idx]  # (baseline_tp, n_elec, n_trials)
        
        # Compute baseline mean and std per electrode across all trials
        self.baseline_means = baseline_data.mean(axis=(0, 2), keepdims=True)  # (1, n_elec, 1)
        self.baseline_stds = baseline_data.std(axis=(0, 2), keepdims=True)   # (1, n_elec, 1)
        
        # Avoid division by zero
        self.baseline_stds[self.baseline_stds == 0] = 1.0
        
        self.stats = {
            "baseline_window_ms": self.baseline_window_ms,
            "baseline_means_shape": self.baseline_means.shape,
            "baseline_stds_shape": self.baseline_stds.shape,
            "mean_baseline": float(self.baseline_means.mean()),
            "mean_baseline_std": float(self.baseline_stds.mean()),
        }
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply baseline normalization and bin."""
        if self.baseline_means is None or self.baseline_stds is None:
            raise RuntimeError("Must call fit() before transform()")
        
        # Apply baseline normalization to raw data
        baseline_normalized = (data - self.baseline_means) / self.baseline_stds
        
        # Now bin the baseline-normalized data
        n_timepoints, n_elec, n_trials = baseline_normalized.shape
        n_bins = n_timepoints // self.bin_size_ms
        
        binned = np.zeros((n_bins, n_elec, n_trials), dtype=np.float32)
        for b in range(n_bins):
            start_idx = b * self.bin_size_ms
            end_idx = start_idx + self.bin_size_ms
            binned[b] = baseline_normalized[start_idx:end_idx].mean(axis=0)
        
        return binned


class OptionC(NormalizationPipeline):
    """
    Option C: 100 ms baseline normalization + per-bin/per-electrode z-scoring (RECOMMENDED).
    
    Two-stage pipeline:
      1. Baseline normalization: subtract and divide by pre-stim baseline per electrode
      2. Per-bin z-scoring: z-score binned baseline-normalized data per (bin, electrode)
         using training trials only
    """
    
    def __init__(self, baseline_window_ms: int = 100, bin_size_ms: int = 20):
        super().__init__(
            name="Option C: Baseline + Per-Bin Z-Score (RECOMMENDED)",
            description="100ms baseline subtraction + per-bin/per-electrode z-scoring"
        )
        self.baseline_window_ms = baseline_window_ms
        self.bin_size_ms = bin_size_ms
        self.baseline_means = None
        self.baseline_stds = None
        self.bin_means = None
        self.bin_stds = None
        self.baseline_normalized_data = None
        self.binned_baseline_normalized_data = None
    
    def fit(self, data: np.ndarray, train_idx: np.ndarray) -> None:
        """
        Stage 1: Compute baseline statistics (all trials)
        Stage 2: Bin baseline-normalized data and compute per-bin stats (training trials)
        """
        # Stage 1: Baseline normalization
        baseline_idx = slice(0, self.baseline_window_ms)
        baseline_data = data[baseline_idx]
        
        self.baseline_means = baseline_data.mean(axis=(0, 2), keepdims=True)
        self.baseline_stds = baseline_data.std(axis=(0, 2), keepdims=True)
        self.baseline_stds[self.baseline_stds == 0] = 1.0
        
        # Apply baseline normalization to raw data
        baseline_normalized = (data - self.baseline_means) / self.baseline_stds
        
        # Stage 2: Bin and compute per-bin stats
        n_timepoints, n_elec, n_trials = baseline_normalized.shape
        n_bins = n_timepoints // self.bin_size_ms
        
        binned = np.zeros((n_bins, n_elec, n_trials), dtype=np.float32)
        for b in range(n_bins):
            start_idx = b * self.bin_size_ms
            end_idx = start_idx + self.bin_size_ms
            binned[b] = baseline_normalized[start_idx:end_idx].mean(axis=0)
        
        self.binned_baseline_normalized_data = binned
        
        # Compute per-bin stats from training trials
        train_binned = binned[:, :, train_idx]
        self.bin_means = train_binned.mean(axis=2, keepdims=True)
        self.bin_stds = train_binned.std(axis=2, keepdims=True)
        self.bin_stds[self.bin_stds == 0] = 1.0
        
        self.stats = {
            "baseline_window_ms": self.baseline_window_ms,
            "bin_size_ms": self.bin_size_ms,
            "baseline_means_shape": self.baseline_means.shape,
            "baseline_stds_shape": self.baseline_stds.shape,
            "bin_means_shape": self.bin_means.shape,
            "bin_stds_shape": self.bin_stds.shape,
            "mean_baseline": float(self.baseline_means.mean()),
            "mean_baseline_std": float(self.baseline_stds.mean()),
            "mean_bin_means": float(self.bin_means.mean()),
            "mean_bin_stds": float(self.bin_stds.mean()),
        }
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply two-stage normalization."""
        if self.baseline_means is None or self.binned_baseline_normalized_data is None:
            raise RuntimeError("Must call fit() before transform()")
        
        normalized = (self.binned_baseline_normalized_data - self.bin_means) / self.bin_stds
        return normalized


# ============================================================================
# Validation & Comparison Utilities
# ============================================================================

class NormalizationValidator:
    """Validate and compare different normalization methods."""
    
    def __init__(self, brain_regions: Dict[str, Tuple[int, int]]):
        self.brain_regions = brain_regions
        self.results = {}
    
    def compare_methods(
        self,
        allmua: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        methods: list,
    ) -> Dict[str, Any]:
        """
        Compare different normalization methods on the same data.
        
        Args:
            allmua: (n_timepoints, n_electrodes, n_trials)
            train_idx: training trial indices
            test_idx: test trial indices
            methods: list of NormalizationPipeline objects
        
        Returns:
            dict with comparison results
        """
        results = {}
        
        for method in methods:
            print(f"\n{'='*70}")
            print(f"Testing: {method.name}")
            print(f"{method.description}")
            print('='*70)
            
            # Fit and transform
            method.fit(allmua, train_idx)
            normalized = method.transform(allmua)
            
            print(f"  Normalized data shape: {normalized.shape}")
            print(f"  Data dtype: {normalized.dtype}")
            print(f"  Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
            print(f"  Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
            
            # Compute sanity checks
            train_data = normalized[:, :, train_idx]
            test_data = normalized[:, :, test_idx]
            
            checks = {
                "train_mean": float(train_data.mean()),
                "train_std": float(train_data.std()),
                "test_mean": float(test_data.mean()),
                "test_std": float(test_data.std()),
                "method_name": method.name,
                "normalized_shape": normalized.shape,
                "stats": method.stats,
                "data": normalized,  # Store full normalized data for downstream analysis
            }
            
            print(f"  Train mean: {checks['train_mean']:.4f}, std: {checks['train_std']:.4f}")
            print(f"  Test mean: {checks['test_mean']:.4f}, std: {checks['test_std']:.4f}")
            
            results[method.name] = checks
        
        return results
    
    def validate_against_original(
        self,
        results: Dict[str, Any],
        original_norm_data: Optional[Dict[str, np.ndarray]],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """
        Validate our normalization methods against the original authors' normMUA.
        
        Args:
            results: dict from compare_methods()
            original_norm_data: dict with original normalized MUA
            train_idx: training trial indices
            test_idx: test trial indices
        """
        if original_norm_data is None:
            print("\nSkipping validation against original (not available).")
            return
        
        print(f"\n{'='*70}")
        print("Validation Against Original THINGSnormMUA")
        print('='*70)
        
        # Extract original data
        if 'normMUA' in original_norm_data:
            orig_norm_mua = original_norm_data['normMUA']
        else:
            # Try other common keys
            norm_keys = [k for k in original_norm_data.keys() if 'norm' in k.lower() and 'mua' in k.lower()]
            if norm_keys:
                orig_norm_mua = original_norm_data[norm_keys[0]]
                print(f"Using key: {norm_keys[0]}")
            else:
                print("Could not find original normalized MUA. Skipping validation.")
                return
        
        print(f"Original normMUA shape: {orig_norm_mua.shape}")
        
        # Compare each method's output to the original
        for method_name, result in results.items():
            normalized = result['data']
            
            # Ensure shape compatibility
            if normalized.shape != orig_norm_mua.shape:
                print(f"\n{method_name}: Shape mismatch ({normalized.shape} vs {orig_norm_mua.shape})")
                continue
            
            # Compute correlation between our method and original
            corr = np.corrcoef(normalized.flatten(), orig_norm_mua.flatten())[0, 1]
            mse = np.mean((normalized - orig_norm_mua) ** 2)
            mae = np.mean(np.abs(normalized - orig_norm_mua))
            
            print(f"\n{method_name}:")
            print(f"  Correlation with original: {corr:.4f}")
            print(f"  MSE vs original: {mse:.6f}")
            print(f"  MAE vs original: {mae:.6f}")


def plot_comparison(
    results: Dict[str, Any],
    brain_regions: Dict[str, Tuple[int, int]],
    region: str = "IT",
    output_dir: str = "./results",
) -> None:
    """
    Plot timecourse comparisons between different normalization methods.
    
    Args:
        results: dict from compare_methods()
        brain_regions: dict mapping region names to electrode ranges
        region: which brain region to plot
        output_dir: where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if region not in brain_regions:
        print(f"Region {region} not in brain_regions. Skipping plots.")
        return
    
    start_elec, end_elec = brain_regions[region]
    n_electrodes = end_elec - start_elec
    
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for ax, (method_name, result) in zip(axes, results.items()):
        normalized = result['data']  # (n_bins, n_electrodes, n_trials)
        
        # Take mean across region and trials for a simple timecourse
        region_data = normalized[:, start_elec:end_elec, :].mean(axis=(1, 2))
        
        ax.plot(region_data, linewidth=2, marker='o')
        ax.set_title(f"{method_name} - {region} Mean Response", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Bin (20 ms bins)")
        ax.set_ylabel("Normalized Response (z-score)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_{region}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare MUA normalization methods for THINGS ventral-stream decoding"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing TVSD .mat files"
    )
    parser.add_argument(
        "--monkey",
        type=str,
        default="monkeyF",
        choices=["monkeyF", "monkeyN"],
        help="Which monkey's data to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Where to save results and plots"
    )
    parser.add_argument(
        "--bin_size_ms",
        type=int,
        default=20,
        help="Temporal bin size in milliseconds"
    )
    parser.add_argument(
        "--baseline_window_ms",
        type=int,
        default=100,
        help="Pre-stimulus baseline window in milliseconds"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Brain region definitions (electrode indices)
    brain_regions = {
        'monkeyF': {
            'V1': (0, 512),
            'IT': (512, 832),
            'V4': (832, 1024)
        },
        'monkeyN': {
            'V1': (0, 512),
            'V4': (512, 768),
            'IT': (768, 1024)
        }
    }
    
    monkey_regions = brain_regions[args.monkey]
    
    # ====== LOAD DATA ======
    print(f"\n{'#'*70}")
    print(f"# Loading TVSD data for {args.monkey}")
    print(f"{'#'*70}\n")
    
    allmua, tb, train_idx, test_idx = load_mua_trials(args.data_dir, args.monkey)
    original_norm_data = load_norm_mua_original(args.data_dir, args.monkey)
    
    # ====== INITIALIZE NORMALIZATION METHODS ======
    print(f"\n{'#'*70}")
    print("# Initializing normalization methods")
    print(f"{'#'*70}\n")
    
    methods = [
        OptionA(bin_size_ms=args.bin_size_ms),
        OptionB(baseline_window_ms=args.baseline_window_ms, bin_size_ms=args.bin_size_ms),
        OptionC(baseline_window_ms=args.baseline_window_ms, bin_size_ms=args.bin_size_ms),
    ]
    
    # ====== RUN COMPARISON ======
    print(f"\n{'#'*70}")
    print("# Comparing normalization methods")
    print(f"{'#'*70}")
    
    validator = NormalizationValidator(monkey_regions)
    results = validator.compare_methods(allmua, train_idx, test_idx, methods)
    
    # ====== VALIDATE AGAINST ORIGINAL ======
    print(f"\n{'#'*70}")
    print("# Validating against original authors' THINGSnormMUA")
    print(f"{'#'*70}")
    
    validator.validate_against_original(results, original_norm_data, train_idx, test_idx)
    
    # ====== SAVE RESULTS ======
    print(f"\n{'#'*70}")
    print("# Saving results")
    print(f"{'#'*70}\n")
    
    # Save statistics summary as JSON
    stats_summary = {}
    for method_name, result in results.items():
        stats_summary[method_name] = {
            'description': result.get('method_name', ''),
            'shape': str(result['normalized_shape']),
            'train_mean': result['train_mean'],
            'train_std': result['train_std'],
            'test_mean': result['test_mean'],
            'test_std': result['test_std'],
            'stats': result['stats'],
        }
    
    stats_path = os.path.join(args.output_dir, "normalization_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    # Save normalized data for each method as .mat files
    for method_name, result in results.items():
        normalized = result['data']
        
        # Create sanitized filename
        sanitized_name = method_name.lower().replace(" ", "_").replace(":", "").replace("+", "_")
        mat_path = os.path.join(args.output_dir, f"{args.monkey}_{sanitized_name}.mat")
        
        savemat(mat_path, {
            'normalized_mua': normalized.astype(np.float32),
            'method_name': method_name,
            'train_indices': train_idx,
            'test_indices': test_idx,
        })
        print(f"Saved normalized MUA to {mat_path}")
    
    # ====== GENERATE PLOTS ======
    print(f"\n{'#'*70}")
    print("# Generating comparison plots")
    print(f"{'#'*70}\n")
    
    for region in monkey_regions.keys():
        plot_comparison(results, monkey_regions, region=region, output_dir=args.output_dir)
    
    print(f"\n{'#'*70}")
    print("# Done!")
    print(f"{'#'*70}\n")
    print(f"Results saved to {args.output_dir}")
    print("\nRECOMMENDATION:")
    print("Use Option C (Baseline + Per-Bin Z-Score) for your visual decoding pipeline.")
    print("This method:")
    print("  - Removes electrode-specific baseline drifts")
    print("  - Standardizes per-bin variance for well-conditioned models")
    print("  - Is compatible with original THINGSnormMUA design")
    print("  - Extends naturally to per-day normalization")


if __name__ == "__main__":
    main()
