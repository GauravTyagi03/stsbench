#!/bin/bash
#
#SBATCH --job-name=dorsal_ddpm_ts
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/train_ddpm_ts.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/train_ddpm_ts.%j.err
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
cd /oak/stanford/groups/anishm/gtyagi/stsbench/timeseries

# Thread settings for CPU operations
N=8
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# PyTorch CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Starting timeseries DDPM training (ventral)..."
python3 train_ddpm_cond_ts.py --config configs/ventral_stream_diffusion_ts.yaml
echo "Timeseries DDPM training completed!"
