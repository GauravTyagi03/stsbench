#!/bin/bash
#
#SBATCH --job-name=ts_preprocess
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/preprocess_ts.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/preprocess_ts.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p anishm
#SBATCH --mem=256G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# ---- environment ----
module purge
module load python/3.12.1
module load hdf5/1.14.4

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate

echo "--- Environment Check ---"
which python
python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
echo "-------------------------"

cd /oak/stanford/groups/anishm/gtyagi/stsbench/timeseries
mkdir -p logs/slurm

# # ---- dorsal stream ----
# echo "Preprocessing dorsal stream timeseries..."
# python preprocess_timeseries.py \
#     --timeseries_h5  /oak/stanford/groups/anishm/gtyagi/stsbench/results/monkeyF_timeseries_normalized.h5 \
#     --pickle_path    /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/dorsal_stream_dataset.pickle \
#     --output_path    /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/dorsal_stream_timeseries_preprocessed.h5 \
#     --num_bins       15
# echo "Dorsal preprocessing done."

# ---- ventral stream ----
echo "Preprocessing ventral stream timeseries..."
python preprocess_timeseries.py \
    --timeseries_h5  /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/monkeyF_timeseries_normalized.h5 \
    --raw_mat        /scratch/groups/anishm/tvsd/monkeyN_THINGS_MUA_trials.mat \
    --output_path    /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/ventral_stream_timeseries_preprocessed.h5 \
    --num_bins       15
echo "Ventral preprocessing done."
