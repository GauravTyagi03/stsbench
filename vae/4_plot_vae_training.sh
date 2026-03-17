#!/bin/bash
#
#SBATCH --job-name=plot_vae_training
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/stsbench/vae/logs/slurm_plot_training.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/stsbench/vae/logs/slurm_plot_training.%j.err
#SBATCH --time=0:30:00
#SBATCH --qos=normal
#SBATCH -p anishm
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# ---- environment ----
module purge
module load python/3.12.1
module load py-numpy/1.26.3_py312

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source /oak/stanford/groups/anishm/gtyagi/stsbench/venv/bin/activate
cd /oak/stanford/groups/anishm/gtyagi/stsbench/vae

# ---- collect all training logs ----
LOG_DIR="/oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction/logs/ventral_stream"
LOG_FILES=("${LOG_DIR}"/vae_*/vae_training_log.txt)

# filter to only existing files (glob may expand to literal string if no match)
EXISTING=()
for f in "${LOG_FILES[@]}"; do
    [[ -f "$f" ]] && EXISTING+=("$f")
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "No vae_training_log.txt files found under ${LOG_DIR}/vae_*/"
    exit 1
fi

echo "Found ${#EXISTING[@]} log file(s):"
for f in "${EXISTING[@]}"; do echo "  $f"; done
echo ""

echo "Plotting training curves..."
python3 plot_vae_training.py --logs "${EXISTING[@]}"
echo "Done!"
