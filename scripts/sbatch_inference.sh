#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# List of file paths and datasets
file_paths=(
# Imagenet-100
   "saved_models/imagenet100/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_CosineAnnealingWarmRestarts/lr_0.001/epsilon_16/max_epsilon_16.pth"
#    "saved_models/imagenet100/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_CosineAnnealingWarmRestarts/lr_0.001/epsilon_8/max_epsilon_8.pth"
#    "saved_models/imagenet100/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_WarmupCosineLR/lr_0.001/max_epsilon_16.pth"
#    "perceptual-advex/imagenet100/pat_alexnet_0.5.pt"
#    "perceptual-advex/imagenet100/pat_self_0.25.pt" 

# Imagenet
    # "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_True/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_CosineAnnealingWarmRestarts/max_epsilon_16.pth"
    # "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_True/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_WarmupCosineLR/lr_0.001/max_epsilon_16.pth"
    # "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_WarmupCosineLR/lr_0.001/max_epsilon_16.pth"
    # "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_adaptive/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/schdeuler_CosineAnnealingWarmRestarts/lr_0.001/max_epsilon_16.pth"
#    "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_re_introduce/agnostic_loss_False/GradAlign_True/optimizer_SGD/pgd_steps_2/max_epsilon_16.pth"
#    "saved_models/imagenet/fine_tune_clean/resnet50/seed_42/train_method_re_introduce/agnostic_loss_True/GradAlign_True/optimizer_SGD/pgd_steps_2/max_epsilon_16.pth"
)

# Initialize node flag
FLAG=1

# Loop through each file path and submit the job
for file in "${file_paths[@]}"; do
    sbatch -c 16 --gres=gpu:1 -w "entropy${FLAG}" -J AA_infer scripts/inference_AutoAttack.sh "$file"
    
    # Alternate the flag between 1 and 2
    if [[ "$FLAG" -eq 1 ]]; then
        FLAG=2
    else
        FLAG=1
    fi

    sleep 1  # Avoid overwhelming Slurm
done
