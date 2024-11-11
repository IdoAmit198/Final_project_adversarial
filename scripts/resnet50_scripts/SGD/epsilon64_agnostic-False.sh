#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 12 --gres=gpu:1 python3 run.py --train_method adaptive --max_epsilon 64 --model_name resnet50 --scheduler WarmupCosineLR --optimizer SGD