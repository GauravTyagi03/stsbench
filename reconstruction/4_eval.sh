#!/bin/bash
#
#SBATCH --job-name=ventral_eval
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/eval.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/eval.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
# 1. Clear everything to avoid "Sherlock Leakage"
module purge
# 2. Load ONLY what is necessary for the interpreter and drivers
module load python/3.12.1
module load cuda/12.4  # Match the cu12 packages in your pip list
module load openblas

# 3. Aggressive path cleaning
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 4. Activate using the full path to the activate script
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate


# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Evaluate reconstructions
echo "Starting evaluation..."
python3 eval.py --config configs/ventral_stream_diffusion.yaml
echo "Evaluation completed!"
