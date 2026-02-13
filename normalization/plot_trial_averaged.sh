#!/bin/bash
#
#SBATCH --job-name=trial_avg_plots
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/trial_avg_plots.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/normalization/logs/slurm/trial_avg_plots.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p anishm
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

# Change to normalization directory
cd /oak/stanford/groups/anishm/gtyagi/stsbench/normalization

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Run trial-averaged plotting for both monkeys
echo "=========================================="
echo "TRIAL-AVERAGED RAW MUA VISUALIZATION"
echo "=========================================="

echo ""
echo "Processing Monkey F..."
python plot_trial_averaged_raw.py \
    --monkey monkeyF \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --output_dir /oak/stanford/groups/anishm/gtyagi/stsbench/results/ \
    --n_electrodes 12

echo ""
echo "Processing Monkey N..."
python plot_trial_averaged_raw.py \
    --monkey monkeyN \
    --data_dir /scratch/groups/anishm/tvsd/ \
    --output_dir /oak/stanford/groups/anishm/gtyagi/stsbench/results/ \
    --n_electrodes 12

echo ""
echo "=========================================="
echo "COMPLETE - Trial-averaged plots created!"
echo "=========================================="
echo "Output location: /oak/stanford/groups/anishm/gtyagi/stsbench/results/"
echo ""
echo "Generated plots:"
echo "  - monkeyF_trial_avg_timeseries.png"
echo "  - monkeyF_trial_avg_heatmap.png"
echo "  - monkeyF_trial_avg_by_region.png"
echo "  - monkeyN_trial_avg_timeseries.png"
echo "  - monkeyN_trial_avg_heatmap.png"
echo "  - monkeyN_trial_avg_by_region.png"
