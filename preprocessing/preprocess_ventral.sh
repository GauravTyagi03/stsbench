#!/bin/bash
#
#SBATCH --job-name=preprocess_ventral
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/preprocess_ventral.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/preprocess_ventral.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
source ~/.bashrc
cd /oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
export PYTHONPATH=""

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Run preprocessing script
python3 preprocess_ventral_dataset.py
