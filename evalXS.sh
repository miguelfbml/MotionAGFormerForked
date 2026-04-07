#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testMotionAGFormerXS    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Starting eval job for MotionAGFormer on MPI-INF-3DHP"


python train_3dhp.py --eval-only --checkpoint checkpoint --checkpoint-file motionagformer-xs-mpi.pth.tr --config configs/mpi/MotionAGFormer-xsmall.yaml