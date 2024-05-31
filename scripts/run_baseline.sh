#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# Use TCMalloc instead of Malloc to boost performance and avoid data leakage.
# export LD_PRELOAD="/usr/lib/libtcmalloc.so" 
# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --train_method train --max_epsilon 32