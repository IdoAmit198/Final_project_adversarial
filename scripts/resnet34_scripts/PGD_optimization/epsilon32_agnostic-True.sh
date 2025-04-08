#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method re_introduce --model_name resnet34 \
    --dataset imagenet100 --Train --Inference --pgd_num_steps 2 --ATAS --agnostic_loss

    