#!/bin/bash
#
#SBATCH --job-name=preprocess_ventral
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/preprocess_ventral.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing/preprocess_ventral.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p anishm
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# # Environment setup
# source ~/.bashrc
# cd /oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing
# source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
# export PYTHONPATH=""

# Environment setup
# 1. Clear everything to avoid "Sherlock Leakage"
module purge
# 2. Load ONLY what is necessary for the interpreter and drivers
module load python/3.12.1
module load cuda/12.4  # Match the cu12 packages in your pip list
module load openblas
module load py-h5py/3.10.0_py312

# 3. Aggressive path cleaning
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 4. Activate using the full path to the activate script
source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
cd /oak/stanford/groups/anishm/gtyagi/stsbench/preprocessing

# Fix for OpenCV on headless compute nodes
export QT_QPA_PLATFORM=offscreen
export OPENCV_IO_ENABLE_OPENEXR=0

# Thread settings for CPU operations
N=4
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# Run preprocessing script
python3 preprocess_ventral_dataset.py
