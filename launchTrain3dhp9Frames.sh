#!/bin/bash
#
#SBATCH --partition=gpu_min24gb     # Reserved partition
#SBATCH --qos=gpu_min24gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=trainMotionAGFormer3dhp9Frames    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Starting training job for MotionAGFormer on MPI-INF-3DHP"

# Set wandb API key if needed
# export WANDB_API_KEY=your_api_key_here

python3 train_3dhp.py \
    --config configs/mpi/MotionAGFormer-small_9.yaml \
    --new-checkpoint checkpoint_mpi_9 \
    --use-wandb \
    --wandb-name "MotionAGFormer_MPI_3DHP_9_frames_training"