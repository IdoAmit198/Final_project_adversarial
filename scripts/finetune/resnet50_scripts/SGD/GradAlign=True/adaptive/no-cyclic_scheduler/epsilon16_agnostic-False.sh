#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method adaptive --model_name timm_resnetv2_50x1_bit.goog_in21k_ft_in1k \
    --dataset imagenet --Train --fine_tune clean --max_epochs 100 \
    --GradAlign --learning_rate 1e-3 --scheduler CosineAnnealingWarmRestarts