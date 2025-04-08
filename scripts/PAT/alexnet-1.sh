#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pat
# Change current directory to "perceptual-advex" directory.
cd perceptual-advex
# Run an evaluation for the LinfAttack with all bound values between 1/255 and 64/255, inclusive.
srun -c 16 --gres=gpu:1  python3 evaluate_trained_model.py --dataset imagenet100 --dataset_path /datasets/ImageNet \
            --checkpoint imagenet100/pat_alexnet_0.5.pt --arch resnet50 --batch_size 64 --output pat_results/imagenet100/alexnet-0.5-evaluation.csv \
            "NoAttack()" \
            "AutoLinfAttack(model, 'imagenet100', bound=1/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=2/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=3/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=4/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=5/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=6/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=7/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=8/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=9/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=10/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=11/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=12/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=13/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=14/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=15/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=16/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=17/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=18/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=19/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=20/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=21/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=22/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=23/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=24/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=25/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=26/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=27/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=28/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=29/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=30/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=31/255)" \
            "AutoLinfAttack(model, 'imagenet100', bound=32/255)" 
            