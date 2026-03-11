#!/bin/bash
#
#SBATCH --job-name=vae_z64_beta01
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta01/slurm_train.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta01/slurm_train.%j.err
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
module load py-pytorch/2.4.1_py312
module load py-torchvision/0.19.1_py312
module load py-pillow/10.2.0_py312
module load py-numpy/1.26.3_py312
module load yaml-cpp/0.7.0

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
cd /oak/stanford/groups/anishm/gtyagi/stsbench/vae

N=8
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z64_beta01
mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/checkpoints/ventral_stream/vae_z64_beta01

echo "Starting VAE training (z64, beta=0.01)..."
python3 train_vae.py --config configs/ventral_vae_z64_beta01.yaml
echo "VAE training complete!"
