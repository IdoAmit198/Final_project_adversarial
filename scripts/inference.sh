#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 16 CPUs!
# srun -c 16 --gres=gpu:1 python3 run.py --model_name robust_bench:Engstrom2019Robustness \
#     --dataset imagenet  --AutoAttackInference \

srun -c 16 --gres=gpu:1 python3 run.py --model_name resnet50 \
    --dataset imagenet100 --AutoAttackInference \
    --eval_model_path saved_models/imagenet100/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_CosineAnnealingWarmRestarts/lr_0.001/epsilon_16/max_epsilon_16.pth