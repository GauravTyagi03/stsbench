#!/bin/bash
#
#SBATCH --job-name=vae_z64_beta001
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta001/slurm_train.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta001/slurm_train.%j.err
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# ---- environment ----
module purge
module load python/3.12.1
module load cuda/12.4
module load hdf5/1.14.4

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
cd /oak/stanford/groups/anishm/gtyagi/stsbench/vae

# Thread settings for CPU operations
N=8
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta001
mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/checkpoints/ventral_stream/vae_z64_beta001

echo "Starting VAE training (z64, beta=0.001)..."
python3 train_vae.py --config configs/ventral_vae_z64_beta001.yaml
echo "VAE training complete!"
