#!/bin/bash
#
#SBATCH --job-name=ventral_eval
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/eval.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/eval.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
source ~/.bashrc
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
export PYTHONPATH=""

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Evaluate reconstructions
echo "Starting evaluation..."
python3 eval.py --config configs/ventral_stream_diffusion.yaml
echo "Evaluation completed!"
