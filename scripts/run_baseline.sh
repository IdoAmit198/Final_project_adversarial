#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# This script requires the sbatch to run with at least 8 CPUs as well!
srun -c 12 --gres=gpu:L40:1 python3 run.py