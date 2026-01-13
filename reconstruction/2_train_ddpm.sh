#!/bin/bash
#
#SBATCH --job-name=ventral_ddpm
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/train_ddpm.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/train_ddpm.%j.err
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH -p owners
#SBATCH -G 1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
source ~/.bashrc
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
export PYTHONPATH=""

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# PyTorch CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Train diffusion model (100 epochs) - requires completed VQ-VAE
echo "Starting DDPM training..."
python3 train_ddpm_cond.py --config configs/ventral_stream_diffusion.yaml
echo "DDPM training completed!"
