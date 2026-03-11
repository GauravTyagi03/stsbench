#!/bin/bash
#
#SBATCH --job-name=plot_vae_recon
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z128_beta001/slurm_plot.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z128_beta001/slurm_plot.%j.err
#SBATCH --time=1:00:00
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
module load py-matplotlib/3.8.2_py312

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

# ---- Edit VAE_CONFIG to point to the config you want to inspect ----
VAE_CONFIG="configs/ventral_vae_z128_beta001.yaml"

echo "Plotting VAE reconstruction check..."
python3 plot_vae_recon.py --config ${VAE_CONFIG} --n_samples 5
echo "Done! See vae_recon_check.png in the config's output_dir."
