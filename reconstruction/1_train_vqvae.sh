#!/bin/bash
#
#SBATCH --job-name=ventral_vqvae
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/train_vqvae.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/train_vqvae.%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=normal
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

# Train VQ-VAE autoencoder (15 epochs)
echo "Starting VQ-VAE training..."
python3 train_vqvae.py --config configs/ventral_stream_diffusion.yaml
echo "VQ-VAE training completed!"
