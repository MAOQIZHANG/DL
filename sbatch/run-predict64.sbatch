#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_11

## change the partition number to use different number of GPUs
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --job-name=p64

module purge

singularity exec --nv \
            --overlay /home/mz3550/pytorch.ext3:ro \
            /home/mz3550/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python -u DL2/predict_22nd_frame64.py > out/predict64.out"
