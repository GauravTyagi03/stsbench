#!/bin/bash
#
# Run eval_vae_diffusion.py after identifying the best VAE checkpoint.
# Edit VAE_CONFIG below to point to the winning config before submitting.
#
#SBATCH --job-name=sample_ddpm_vae
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_sample/slurm_sample.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_sample/slurm_sample.%j.err
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
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

N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Edit this to point to the best VAE config after sweep ----
VAE_CONFIG="configs/ventral_vae_z64_beta001.yaml"
DIFFUSION_CONFIG="../timeseries/configs/ventral_stream_diffusion_ts_conv1d.yaml"

mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_sample

echo "Sampling from DDPM with VAE-decoded conditioning..."
python3 eval_vae_diffusion.py \
    --vae_config       ${VAE_CONFIG} \
    --diffusion_config ${DIFFUSION_CONFIG}
echo "Sampling complete!"
