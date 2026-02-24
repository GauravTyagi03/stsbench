#!/bin/bash
# Master script to submit the full timeseries training pipeline.
# Step 0 (preprocessing) only needs to run once. After the preprocessed HDF5
# exists, comment out JOB0 and start from JOB2.
#
# Dependency chain:
#   preprocess -> train DDPM -> sample -> eval
#
# Note: VQ-VAE training is NOT included here — reuse the existing checkpoint
#       from reconstruction/checkpoints/{stream}/diffusion/vqvae_autoencoder_ckpt.pth

cd /oak/stanford/groups/anishm/gtyagi/stsbench/timeseries
mkdir -p logs/slurm

echo "Submitting timeseries pipeline (dorsal stream)..."
echo "================================================"

# Step 0: Preprocess timeseries data (run once; skip once HDF5 exists)
JOB0=$(sbatch --parsable 0_preprocess_ts.sh)
echo "Job 0 submitted: Timeseries preprocessing (Job ID: $JOB0)"

# Step 2: Train timeseries DDPM — depends on preprocessing
JOB2=$(sbatch --parsable --dependency=afterok:$JOB0 2_train_ddpm_ts.sh)
echo "Job 2 submitted: Timeseries DDPM training (Job ID: $JOB2, depends on $JOB0)"

# Step 3: Sample — depends on DDPM training
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 3_sample_ddpm_ts.sh)
echo "Job 3 submitted: Sampling (Job ID: $JOB3, depends on $JOB2)"

# Step 4: Evaluate reconstructions — depends on sampling
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 4_eval_ts.sh)
echo "Job 4 submitted: Evaluation (Job ID: $JOB4, depends on $JOB3)"

echo "================================================"
echo "All jobs submitted!"
echo ""
echo "Job IDs:"
echo "  Preprocess: $JOB0"
echo "  DDPM train: $JOB2"
echo "  Sampling:   $JOB3"
echo "  Evaluation: $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Cancel all:   scancel $JOB0 $JOB2 $JOB3 $JOB4"
echo ""
echo "To skip preprocessing (HDF5 already exists), edit this script and run:"
echo "  sbatch 2_train_ddpm_ts.sh"
