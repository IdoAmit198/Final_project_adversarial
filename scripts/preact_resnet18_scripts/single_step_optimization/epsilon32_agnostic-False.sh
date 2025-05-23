#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 14 --gres=gpu:1 python3 run.py --train_method re_introduce --model_name preact_resnet18 --dataset imagenet100 --Train --Inference