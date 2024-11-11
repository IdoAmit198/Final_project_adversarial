#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# Use TCMalloc instead of Malloc to boost performance and avoid data leakage.
# export LD_PRELOAD="/usr/lib/libtcmalloc.so" 
# This script requires the sbatch to run with at least 12 CPUs!
srun -c 12 --gres=gpu:1 python3 run.py --train_method re_introduce --re_introduce_prob 0 --sanity_check --max_epsilon 32 