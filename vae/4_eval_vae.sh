#!/bin/bash
#
# Evaluate VAE-conditioned reconstructions (PSNR + LPIPS).
# Set IMAGE_DIR and RUN_NAME below to point at the sampling output folder.
#
#SBATCH --job-name=vae_eval
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/vae/logs/slurm_vae_eval.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/vae/logs/slurm_vae_eval.%j.err
#SBATCH --time=1:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# ---- environment ----
module purge
module load python/3.12.1
module load cuda/12.4
module load openblas
module load py-pytorch/2.4.1_py312
module load py-torchvision/0.19.1_py312
module load py-pillow/10.2.0_py312
module load py-numpy/1.26.3_py312

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

EVAL_OUTPUT=/oak/stanford/groups/anishm/gtyagi/stsbench/vae/logs/eval
mkdir -p ${EVAL_OUTPUT}

# ---- Edit these two lines after the sampling job completes ----
# IMAGE_DIR: full path to the folder containing {i}_pred.png / {i}_true.png
# RUN_NAME:  short label used for output .npy and summary file names

# Timeseries DDPM + VAE (conv1d_k3_skip + wide_nb2)
IMAGE_DIR=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/diffusion_ts_conv1d_k3_skip/vae_conditioned_vae_z128_wide_nb2_JOBID
RUN_NAME=vae_wide_nb2_ts_k3_skip

echo "Evaluating: ${RUN_NAME}"
python3 eval_vae_recon.py \
    --image_dir  ${IMAGE_DIR} \
    --run_name   ${RUN_NAME} \
    --output_dir ${EVAL_OUTPUT}

# Original DDPM + VAE mean (wide_nb2)
IMAGE_DIR=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/diffusion/vae_mean_conditioned_vae_z128_wide_nb2_JOBID
RUN_NAME=vae_wide_nb2_orig_ddpm

echo "Evaluating: ${RUN_NAME}"
python3 eval_vae_recon.py \
    --image_dir  ${IMAGE_DIR} \
    --run_name   ${RUN_NAME} \
    --output_dir ${EVAL_OUTPUT}

echo "All evaluations complete!"
