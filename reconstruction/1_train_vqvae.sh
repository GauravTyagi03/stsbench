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
#source ~/.bashrc
module purge
module load python/3.12.1
module load cuda/11.5.0
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

ENV=/oak/stanford/groups/anishm/gtyagi/stsbench/venv
export VIRTUAL_ENV=$VENV
export PATH="$VENV/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# PyTorch CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Sanity check (keep until stable) ----
$VENV/bin/python - <<EOF
import sys, numpy
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("NumPy path:", numpy.__file__)
EOF

# Train VQ-VAE autoencoder (15 epochs)
echo "Starting VQ-VAE training..."
python train_vqvae.py --config configs/ventral_stream_diffusion.yaml
echo "VQ-VAE training completed!"
