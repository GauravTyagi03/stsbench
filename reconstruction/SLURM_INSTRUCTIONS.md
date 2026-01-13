# Ventral Stream Training and Evaluation - SLURM Instructions

## Overview
This directory contains SLURM scripts for training and evaluating ventral stream reconstruction models using diffusion models.

## Pipeline Stages

### 1. Train VQ-VAE (`1_train_vqvae.sh`)
- **Time**: ~12 hours
- **GPU**: 1x A100
- **Memory**: 32GB
- Trains the VQ-VAE autoencoder (15 epochs)

### 2. Train DDPM (`2_train_ddpm.sh`)
- **Time**: ~48 hours
- **GPU**: 1x A100
- **Memory**: 32GB
- **QOS**: long (requires long partition access)
- Trains the diffusion model (100 epochs)
- **Depends on**: VQ-VAE checkpoint from step 1

### 3. Sample from Model (`3_sample_ddpm.sh`)
- **Time**: ~4 hours
- **GPU**: 1x A100
- **Memory**: 16GB
- Generates reconstructions from trained model
- **Depends on**: DDPM checkpoint from step 2

### 4. Evaluate (`4_eval.sh`)
- **Time**: ~2 hours
- **GPU**: 1x A100
- **Memory**: 16GB
- Computes metrics (LPIPS, PSNR) on reconstructions
- **Depends on**: Samples from step 3

## Usage Options

### Option A: Submit All Jobs with Dependencies (Recommended)
```bash
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction
bash run_all_ventral.sh
```

This submits all 4 jobs with proper dependencies. Jobs will automatically start when the previous job completes successfully.

### Option B: Submit Jobs Individually
```bash
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

# Submit step 1
sbatch 1_train_vqvae.sh

# After step 1 completes, submit step 2
sbatch 2_train_ddpm.sh

# After step 2 completes, submit step 3
sbatch 3_sample_ddpm.sh

# After step 3 completes, submit step 4
sbatch 4_eval.sh
```

### Option C: Submit with Manual Dependencies
```bash
JOB1=$(sbatch --parsable 1_train_vqvae.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 2_train_ddpm.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 3_sample_ddpm.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 4_eval.sh)
```

## Monitoring

### Check job status
```bash
squeue -u gtyagi
```

### Check specific job details
```bash
scontrol show job <JOB_ID>
```

### View output logs (in real-time)
```bash
tail -f logs/train_vqvae.<JOB_ID>.out
tail -f logs/train_ddpm.<JOB_ID>.out
tail -f logs/sample_ddpm.<JOB_ID>.out
tail -f logs/eval.<JOB_ID>.out
```

### Cancel jobs
```bash
# Cancel single job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u gtyagi
```

## Output Locations

Based on `configs/ventral_stream_diffusion.yaml`:

- **Checkpoints**: `./checkpoints/ventral_stream/diffusion/`
  - `vqvae_autoencoder_ckpt.pth`
  - `ddpm_ckpt_neural_cond.pth`

- **Logs**: `./logs/ventral_stream/diffusion/`
  - `vqvae_training_log.txt`
  - `ldm_training_log.txt`
  - Sample images during training

- **SLURM logs**: `./logs/`
  - `train_vqvae.<JOB_ID>.out/err`
  - `train_ddpm.<JOB_ID>.out/err`
  - `sample_ddpm.<JOB_ID>.out/err`
  - `eval.<JOB_ID>.out/err`

## Total Estimated Time
- **Sequential**: ~66 hours (~2.75 days)
- Jobs run one after another automatically with dependencies

## Important Notes

1. **Long QOS**: The DDPM training job uses `--qos=long` (48 hours). Make sure you have access to the long partition. If not, you may need to:
   - Request access from your cluster admin
   - OR split the training into multiple checkpointed runs
   - OR reduce the number of epochs in the config

2. **GPU Requirements**: All jobs request A100 GPUs. If these aren't available, you can modify the scripts to:
   - Remove the `-C GPU_SKU:A100_SXM4` line to use any available GPU
   - OR change to a different GPU type

3. **Dependencies**: The `run_all_ventral.sh` script uses `--dependency=afterok:<JOB_ID>` which means:
   - Next job only starts if previous job completes successfully
   - If a job fails, subsequent jobs won't start
   - You'll receive email notifications when jobs fail

4. **Restarting**: If a job fails:
   - Check the error logs in `logs/`
   - Fix the issue
   - Resubmit from that step forward

## Example Workflow

```bash
# Navigate to reconstruction directory
cd /oak/stanford/groups/anishm/gtyagi/stsbench/reconstruction

# Submit all jobs
bash run_all_ventral.sh

# Output will show:
# Job 1 submitted: VQ-VAE training (Job ID: 12345)
# Job 2 submitted: DDPM training (Job ID: 12346, depends on 12345)
# Job 3 submitted: Sampling (Job ID: 12347, depends on 12346)
# Job 4 submitted: Evaluation (Job ID: 12348, depends on 12347)

# Monitor progress
squeue -u gtyagi

# Watch current job output
tail -f logs/train_vqvae.12345.out
```
