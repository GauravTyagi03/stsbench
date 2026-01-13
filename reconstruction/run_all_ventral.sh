#!/bin/bash
# Master script to submit all ventral stream training and evaluation jobs
# Each job depends on the previous one completing successfully

cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Submitting ventral stream training pipeline..."
echo "================================================"

# Step 1: Train VQ-VAE autoencoder (15 epochs, ~12 hours)
JOB1=$(sbatch --parsable 1_train_vqvae.sh)
echo "Job 1 submitted: VQ-VAE training (Job ID: $JOB1)"

# Step 2: Train DDPM conditioned model (100 epochs, ~48 hours) - depends on Job 1
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 2_train_ddpm.sh)
echo "Job 2 submitted: DDPM training (Job ID: $JOB2, depends on $JOB1)"

# Step 3: Sample from trained model (~4 hours) - depends on Job 2
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 3_sample_ddpm.sh)
echo "Job 3 submitted: Sampling (Job ID: $JOB3, depends on $JOB2)"

# Step 4: Evaluate reconstructions (~2 hours) - depends on Job 3
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 4_eval.sh)
echo "Job 4 submitted: Evaluation (Job ID: $JOB4, depends on $JOB3)"

echo "================================================"
echo "All jobs submitted successfully!"
echo ""
echo "Job IDs:"
echo "  VQ-VAE:     $JOB1"
echo "  DDPM:       $JOB2"
echo "  Sampling:   $JOB3"
echo "  Evaluation: $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Cancel all: scancel $JOB1 $JOB2 $JOB3 $JOB4"
