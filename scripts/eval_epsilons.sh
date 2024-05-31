#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# Use TCMalloc instead of Malloc to boost performance and avoid data leakage.
# export LD_PRELOAD="/usr/lib/libtcmalloc.so" 
# This script requires the sbatch to run with at least 12 CPUs!
srun -c 20 --gres=gpu:1 python3 run.py --eval_epsilons --eval_uncertainty \
            --eval_model_path saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_32.pth

# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_False/max_epsilon_8.pth # done!
# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_False/max_epsilon_16.pth # done
# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_False/max_epsilon_32.pth # done!

# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_True/max_epsilon_8.pth # done!
# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_True/max_epsilon_16.pth # done!
# saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_True/max_epsilon_32.pth *

# ---------------------------------------------------------------------------------------- #

# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_False/max_epsilon_8.pth # done!
# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_False/max_epsilon_16.pth # done!
# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_False/max_epsilon_32.pth # done!

# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_8.pth # done!
# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_16.pth # done!
# saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/max_epsilon_32.pth *

# ---------------------------------------------------------------------------------------- #

# saved_models/resnet18/seed_42/train_method_train/agnostic_loss_False/max_epsilon_8.pth # done!
# saved_models/resnet18/seed_42/train_method_train/agnostic_loss_False/max_epsilon_16.pth # done!
# saved_models/resnet18/seed_42/train_method_train/agnostic_loss_False/max_epsilon_32.pth # done!