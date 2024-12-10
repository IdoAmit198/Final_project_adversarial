#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 12 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method adaptive --max_epsilon 32 --agnostic_loss --model_name preact_resnet18 --pgd_num_steps 2 --scheduler CyclicLR