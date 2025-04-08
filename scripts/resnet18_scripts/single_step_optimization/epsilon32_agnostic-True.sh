#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 12 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method re_introduce --agnostic_loss --model_name resnet18 \
    --dataset imagenet100 --Train --Inference --ATAS