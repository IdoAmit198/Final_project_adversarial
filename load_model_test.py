from utils.args import get_args
import torch
from torch import nn
import torchvision
from utils.data import load_dataloaders
from adv_train import adv_training
import warnings

import wandb
import random
from datetime import datetime, timedelta
import pytz


model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', weights=None)
num_classes=10
    # if args.data =='cifar10':
    #     num_classes=10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('/home/ido.amit/adversarial_course/Final_project_adversarial/saved_models/wide_resnet50_2/seed_42/train_method_adaptive/max_epsilon_8.pth'))
for p in model.parameters():
    # Verify whether all paramters in model are 0
    print(f"non zero elements:{p.count_nonzero()}")
    print(p)