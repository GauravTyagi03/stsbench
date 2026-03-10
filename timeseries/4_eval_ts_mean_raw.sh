#!/bin/bash
#
#SBATCH --job-name=dorsal_eval_ts_mean_raw
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/eval_ts_mean_raw.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/eval_ts_mean_raw.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p anishm
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
module load hdf5/1.14.4

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate

cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

echo "Starting evaluation of timeseries reconstructions (mean_raw)..."
python3 eval.py --config /oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/configs/ventral_stream_diffusion_ts_mean_raw.yaml
echo "Evaluation completed!"
