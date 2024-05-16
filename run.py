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

if __name__ == '__main__':
    print("Started")

    args = get_args(description='Adversarial training')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.log_name = f'{args.model_name}_train method_{args.train_method}_agnostic_loss_{args.agnostic_loss}_seed_{args.seed}_max epsilon_{int(args.max_epsilon*255)}'
    timezone = pytz.timezone('Asia/Jerusalem')
    args.date_stamp = datetime.now(timezone).strftime("%d/%m_%H:%M")
    wandb.init(project="Adversarial-Project", name=f'{args.date_stamp}_{args.log_name}', entity = "deep_learning_hw4", config=args)
    wandb.define_metric("step")
    wandb.define_metric("Epoch")
    wandb.define_metric("Train epochs loss", step_metric="Epoch")
    wandb.define_metric("Train epochs accuracy", step_metric="Epoch")
    wandb.define_metric("Test epochs accuracy", step_metric="Epoch")
    wandb.define_metric("Epsilons_metrics/min_epsilon", step_metric="Epoch")
    wandb.define_metric("Epsilons_metrics/max_epsilon", step_metric="Epoch")
    wandb.define_metric("Epsilons_metrics/mean_epsilon", step_metric="Epoch")
    wandb.define_metric("Epsilons_metrics/re_introduce_cur_prob", step_metric="Epoch")
    wandb.define_metric("Train lr", step_metric="Epoch")

    num_classes=10
    model = torchvision.models.get_model(args.model_name,num_classes=num_classes, weights=None)
    model = model.to(args.device)

    train_loader, test_loader = load_dataloaders(args.batch_size)

    # scaler = torch.cuda.amp.GradScaler()
    adv_training(model, train_loader, test_loader, args)

    wandb.finish()
    exit
    # run_adv_training(args)