#!/bin/bash
#
#SBATCH --job-name=visualize_norm
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/logs/slurm/visualize_norm.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/logs/slurm/visualize_norm.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
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
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Change to preprocessing directory
cd /oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Run normalization visualization script
echo "Starting normalization visualization..."
python visualize_normalization.py \
    --monkey monkeyF \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --results_dir /oak/stanford/groups/anishm/gtyagi/stsbench/results/ \
    --output_dir /oak/stanford/groups/anishm/gtyagi/stsbench/results/ \
    --n_trials 10 \
    --seed 42

echo "Normalization visualization completed!"
