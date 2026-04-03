#!/bin/bash
#
#SBATCH --job-name=ts_preprocess_nb30
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/preprocess_ts_nb30.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/timeseries/logs/slurm/preprocess_ts_nb30.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p owners
#SBATCH --mem=64G
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

# NOTE: alternative_timeseries_norm.sh does NOT need to be re-run.
# That step outputs at timepoint granularity — the 30-bin split happens here only.

echo "Preprocessing ventral stream timeseries (30 bins)..."
python preprocess_timeseries.py \
    --timeseries_h5_N /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/monkeyN_timeseries_normalized_no_baseline.h5 \
    --timeseries_h5_F /oak/stanford/groups/anishm/gtyagi/stsbench/normalization/results/monkeyF_timeseries_normalized_no_baseline.h5 \
    --raw_mat_N       /scratch/groups/anishm/tvsd/monkeyN_THINGS_MUA_trials.mat \
    --raw_mat_F       /scratch/groups/anishm/tvsd/monkeyF_THINGS_MUA_trials.mat \
    --paper_norm_N    /oak/stanford/groups/anishm/gtyagi/stsbench/results/monkeyN_paper_normalized.mat \
    --paper_norm_F    /oak/stanford/groups/anishm/gtyagi/stsbench/results/monkeyF_paper_normalized.mat \
    --output_path     /oak/stanford/groups/anishm/gtyagi/stsbench/dataset/ventral_stream_timeseries_preprocessed_nb30_no_baseline.h5 \
    --num_bins        30
echo "Ventral preprocessing (30 bins) done."
