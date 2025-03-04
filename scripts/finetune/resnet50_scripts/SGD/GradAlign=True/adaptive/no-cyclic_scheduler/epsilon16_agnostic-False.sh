#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method adaptive --model_name resnet50 \
    --dataset imagenet --Train --Inference --fine_tune clean --max_epochs 50 \
    --GradAlign --learning_rate 1e-3