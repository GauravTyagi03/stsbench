#!/bin/bash
#
#SBATCH --job-name=dorsal_eval_ts_conv1d_k3_skip_nb30
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/eval_ts_conv1d_k3_skip_nb30.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/eval_ts_conv1d_k3_skip_nb30.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# ---- environment ----
module purge
module load python/3.12.1
module load cuda/12.4
module load openblas

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate

# eval.py must run from reconstruction/ so its relative ./logs/ path resolves correctly.
# model_name: 'diffusion_ts_conv1d_k3_skip_nb30' causes eval.py to read images from
# ./logs/ventral_stream/diffusion_ts_conv1d_k3_skip_nb30/ — matching output_dir in the config.
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

echo "Starting evaluation of timeseries reconstructions (conv1d_k3_skip, nb30)..."
python3 eval.py --config /oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/configs/nb30/ventral_stream_diffusion_ts_conv1d_k3_skip_nb30.yaml
echo "Evaluation completed!"
