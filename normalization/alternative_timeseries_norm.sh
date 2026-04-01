#!/bin/bash
#
#SBATCH --job-name=timeseries_norm_no_baseline
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/timeseries_norm_no_baseline.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/timeseries_norm_no_baseline.%j.err
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
cd /oak/stanford/groups/anishm/gtyagi/stsbench/normalization

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Run alternative time-series normalization (no pre-stimulus baseline subtraction)
# Outputs: monkeyN_timeseries_normalized_no_baseline.h5
#          monkeyF_timeseries_normalized_no_baseline.h5
OUTPUT_DIR=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/

echo "Starting normalization (no baseline) for monkeyN..."
python alternative_timeseries_norm.py \
    --monkey monkeyN \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --output_dir ${OUTPUT_DIR} \
    --bin_width 10 \
    --no_baseline_norm
echo "monkeyN done."

echo "Starting normalization (no baseline) for monkeyF..."
python alternative_timeseries_norm.py \
    --monkey monkeyF \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --output_dir ${OUTPUT_DIR} \
    --bin_width 10 \
    --no_baseline_norm
echo "monkeyF done."

echo "Alternative time-series normalization (no baseline) completed!"
