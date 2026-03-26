#!/bin/bash
#
#SBATCH --job-name=gen_vaerecon_e200
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z128_beta001_e200/slurm_gen_recon_ts.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z128_beta001_e200/slurm_gen_recon_ts.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=32G
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
module load py-numpy/1.26.3_py312

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
cd /oak/stanford/groups/anishm/gtyagi/stsbench/vae

mkdir -p /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream/vae_z128_beta001_e200

echo "Generating VAE-reconstructed timeseries HDF5 (vae_z128_beta001_e200)..."
python3 generate_vae_recon_ts.py \
    --vae_config configs/ventral_vae_z128_beta001_e200.yaml \
    --output_h5  /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/ventral_stream_timeseries_vaerecon_beta001_e200.h5 \
    --batch_size 256
echo "HDF5 generation complete!"
