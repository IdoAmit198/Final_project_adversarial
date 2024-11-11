#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pat
# Change current directory to "perceptual-advex" directory.
cd perceptual-advex
# Run an evaluation for the LinfAttack with all bound values between 1/255 and 64/255, inclusive.
srun -c 12 --gres=gpu:1 python3 evaluate_trained_model.py --dataset cifar \
            --checkpoint cifar/pat_alexnet_1.pt --arch resnet50 --batch_size 32 --output pat_results/alexnet-1/evaluation.csv \
            "AutoLinfAttack(model, 'cifar', bound=1/255)" \
            "AutoLinfAttack(model, 'cifar', bound=2/255)" \
            "AutoLinfAttack(model, 'cifar', bound=3/255)" \
            "AutoLinfAttack(model, 'cifar', bound=4/255)" \
            "AutoLinfAttack(model, 'cifar', bound=5/255)" \
            "AutoLinfAttack(model, 'cifar', bound=6/255)" \
            "AutoLinfAttack(model, 'cifar', bound=7/255)" \
            "AutoLinfAttack(model, 'cifar', bound=8/255)" \
            "AutoLinfAttack(model, 'cifar', bound=9/255)" \
            "AutoLinfAttack(model, 'cifar', bound=10/255)" \
            "AutoLinfAttack(model, 'cifar', bound=11/255)" \
            "AutoLinfAttack(model, 'cifar', bound=12/255)" \
            "AutoLinfAttack(model, 'cifar', bound=13/255)" \
            "AutoLinfAttack(model, 'cifar', bound=14/255)" \
            "AutoLinfAttack(model, 'cifar', bound=15/255)" \
            "AutoLinfAttack(model, 'cifar', bound=16/255)" \
            "AutoLinfAttack(model, 'cifar', bound=17/255)" \
            "AutoLinfAttack(model, 'cifar', bound=18/255)" \
            "AutoLinfAttack(model, 'cifar', bound=19/255)" \
            "AutoLinfAttack(model, 'cifar', bound=20/255)" \
            "AutoLinfAttack(model, 'cifar', bound=21/255)" \
            "AutoLinfAttack(model, 'cifar', bound=22/255)" \
            "AutoLinfAttack(model, 'cifar', bound=23/255)" \
            "AutoLinfAttack(model, 'cifar', bound=24/255)" \
            "AutoLinfAttack(model, 'cifar', bound=25/255)" \
            "AutoLinfAttack(model, 'cifar', bound=26/255)" \
            "AutoLinfAttack(model, 'cifar', bound=27/255)" \
            "AutoLinfAttack(model, 'cifar', bound=28/255)" \
            "AutoLinfAttack(model, 'cifar', bound=29/255)" \
            "AutoLinfAttack(model, 'cifar', bound=30/255)" \
            "AutoLinfAttack(model, 'cifar', bound=31/255)" \
            "AutoLinfAttack(model, 'cifar', bound=32/255)" \
            "AutoLinfAttack(model, 'cifar', bound=33/255)" \
            "AutoLinfAttack(model, 'cifar', bound=34/255)" \
            "AutoLinfAttack(model, 'cifar', bound=35/255)" \
            "AutoLinfAttack(model, 'cifar', bound=36/255)" \
            "AutoLinfAttack(model, 'cifar', bound=37/255)" \
            "AutoLinfAttack(model, 'cifar', bound=38/255)" \
            "AutoLinfAttack(model, 'cifar', bound=39/255)" \
            "AutoLinfAttack(model, 'cifar', bound=40/255)" \
            "AutoLinfAttack(model, 'cifar', bound=41/255)" \
            "AutoLinfAttack(model, 'cifar', bound=42/255)" \
            "AutoLinfAttack(model, 'cifar', bound=43/255)" \
            "AutoLinfAttack(model, 'cifar', bound=44/255)" \
            "AutoLinfAttack(model, 'cifar', bound=45/255)" \
            "AutoLinfAttack(model, 'cifar', bound=46/255)" \
            "AutoLinfAttack(model, 'cifar', bound=47/255)" \
            "AutoLinfAttack(model, 'cifar', bound=48/255)" \
            "AutoLinfAttack(model, 'cifar', bound=49/255)" \
            "AutoLinfAttack(model, 'cifar', bound=50/255)" \
            "AutoLinfAttack(model, 'cifar', bound=51/255)" "AutoLinfAttack(model, 'cifar', bound=52/255)" \
            "AutoLinfAttack(model, 'cifar', bound=53/255)" "AutoLinfAttack(model, 'cifar', bound=54/255)" \
            "AutoLinfAttack(model, 'cifar', bound=55/255)" "AutoLinfAttack(model, 'cifar', bound=56/255)" \
            "AutoLinfAttack(model, 'cifar', bound=57/255)" "AutoLinfAttack(model, 'cifar', bound=58/255)" \
            "AutoLinfAttack(model, 'cifar', bound=59/255)" "AutoLinfAttack(model, 'cifar', bound=60/255)" \
            "AutoLinfAttack(model, 'cifar', bound=61/255)" "AutoLinfAttack(model, 'cifar', bound=62/255)" \
            "AutoLinfAttack(model, 'cifar', bound=63/255)" "AutoLinfAttack(model, 'cifar', bound=64/255)" 

