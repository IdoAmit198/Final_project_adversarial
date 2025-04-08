#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --model_name robust_bench:Wong2020Fast \
    --dataset imagenet  --AutoAttackInference \