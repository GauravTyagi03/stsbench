#!/bin/bash
#
#SBATCH --job-name=debug_visualize
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/debug_visualize.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/debug_visualize.%j.err
#SBATCH --time=1:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# 1. Clear everything to avoid "Sherlock Leakage"
module purge
# 2. Load ONLY what is necessary for the interpreter and drivers
module load python/3.12.1
module load hdf5/1.14.4

# 3. Aggressive path cleaning
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 4. Activate using the full path to the activate script
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate

# 5. Debug check (Optional but recommended for the first run)
echo "--- Environment Check ---"
which python
python --version
echo "HDF5 module loaded"
echo "-------------------------"

# Thread settings for CPU operations
N=2
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Change to normalization directory
cd /oak/stanford/groups/anishm/gtyagi/stsbench/normalization

# Create logs and output directories if they don't exist
mkdir -p logs/slurm
mkdir -p debug_plots

# Run debug visualization script
echo "Starting normalization debugging visualization..."
python debug_visualize.py \
    --monkey monkeyF \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --results_dir /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/ \
    --output_dir /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/debug_plots/ \
    --bin_width 10

echo "Debug visualization completed!"
echo "Plots saved to: /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/debug_plots/"
