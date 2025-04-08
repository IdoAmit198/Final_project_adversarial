#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236207

# Read arguments
MODEL_PATH=$1

# Extract the dataset name (exact matching order)
if [[ "$MODEL_PATH" == *"imagenet100"* ]]; then
    DATASET="imagenet100"
elif [[ "$MODEL_PATH" == *"imagenet"* ]]; then
    DATASET="imagenet"
elif [[ "$MODEL_PATH" == *"cifar10"* ]]; then
    DATASET="cifar10"
else
    echo "Error: Dataset name could not be determined from path: $FILE_PATH"
    exit 1
fi

echo "Processing file: $FILE_PATH"
echo "Detected dataset: $DATASET"

# This script requires the sbatch to run with at least 16 CPUs!
srun -c 16 --gres=gpu:1 python3 run.py --model_name resnet50 \
    --dataset "$DATASET"  --AutoAttackInference  \
    --eval_model_path "$MODEL_PATH"