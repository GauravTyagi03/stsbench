#!/bin/bash
#
#SBATCH --job-name=compare_norm_methods
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/plotting/logs/slurm/compare_norm_methods.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/plotting/logs/slurm/compare_norm_methods.%j.err
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# 1. Clear everything to avoid "Sherlock Leakage"
module purge
# 2. Load ONLY what is necessary for the interpreter and drivers
module load python/3.12.1

# 3. Aggressive path cleaning
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 4. Activate using the full path to the activate script
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate

# 5. Debug check (Optional but recommended for the first run)
echo "--- Environment Check ---"
which python
python --version
echo "-------------------------"

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Change to plotting directory
cd /oak/stanford/groups/anishm/gtyagi/stsbench/plotting

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Run normalization comparison script
# Update these paths as needed for your data location
echo "Starting normalization methods comparison..."
python compare_norm_methods.py \
    --data_dir /oak/stanford/groups/anishm/gtyagi/stsbench/data \
    --monkey monkeyF \
    --output_dir ./results/compare_norm_methods \
    --bin_size_ms 20 \
    --baseline_window_ms 100
echo "Normalization comparison completed!"