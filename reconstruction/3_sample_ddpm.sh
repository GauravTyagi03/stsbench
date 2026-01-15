#!/bin/bash
#
#SBATCH --job-name=ventral_sample
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/sample_ddpm.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/sample_ddpm.%j.err
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH -C GPU_SKU:A100_SXM4
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

# PyTorch CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Sample from trained diffusion model
echo "Starting sampling from trained model..."
python3 sample_ddpm_cond.py --config configs/ventral_stream_diffusion.yaml
echo "Sampling completed!"
