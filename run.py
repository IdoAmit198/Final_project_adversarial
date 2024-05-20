from utils.args import get_args
import torch
from torch import nn
import torchvision
from utils.data import load_dataloaders
from adv_train import adv_training, adv_eval
import warnings

import pandas as pd
import re

import wandb
import random
from datetime import datetime, timedelta
import pytz

if __name__ == '__main__':
    print("Started")

    args = get_args(description='Adversarial training')
    args.max_epsilon = args.max_epsilon/255
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    
    train_loader, test_loader = load_dataloaders(args.batch_size)
    num_classes=10
    model = torchvision.models.get_model(args.model_name,num_classes=num_classes, weights=None)
    model = model.to(args.device)

    # scaler = torch.cuda.amp.GradScaler()
    if not args.eval_epsilons:
        # wandb login and logs
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
        adv_training(model, train_loader, test_loader, args)
        wandb.finish()
    
    else:
        # load statce_dict of the model from given path
        model.load_state_dict(torch.load(args.eval_model_path))
        print("Model loaded")
        print(f"path: {args.eval_model_path}")
        # Extract the eval_model_ath dir out of the args.eval_model_path path, by ignoring the pth file at the suffix
        save_dir = args.eval_model_path[:args.eval_model_path.rfind('/')]
        print(f"save_dir: {save_dir}")
        eval_trained_epsilon = args.eval_model_path.split('/')[-1]
        eval_trained_epsilon = re.search('[\d][\d]?', eval_trained_epsilon).group()
        assert type(eval_trained_epsilon) == str
        eval_results = []
        train_results = []
        epsilons_list = list(range(args.eval_epsilon_max+1))
        for epsilon in epsilons_list:
            eval_results.append(adv_eval(model, test_loader, args, epsilon/255))
            train_results.append(adv_eval(model, train_loader, args, epsilon/255))
            print(f"Evaluated epsilon:{epsilon} , Test Accuracy: {eval_results[-1]*100}% , Train Accuracy: {train_results[-1]*100}%")
        # Create a pandas dataframe out of the three lists: eval_results, train_results, epsilons_list
        df = pd.DataFrame(list(zip(epsilons_list, eval_results, train_results)), columns=['epsilon', 'eval_results', 'train_results'])
        df.to_csv(f'{save_dir}/eval_accuracy_{eval_trained_epsilon}.csv', index=False)
    
    exit
    # run_adv_training(args)