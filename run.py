import os
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

    if not args.eval_epsilons:
        # wanbd logging initialization
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
        # Actual training
        adv_training(model, train_loader, test_loader, args)
        wandb.finish()
    
    else:
        # load statce_dict of the trained model from given path
        model.load_state_dict(torch.load(args.eval_model_path))
        print("Model loaded")
        print(f"path: {args.eval_model_path}")
        # Extract the eval_model_ath dir out of the args.eval_model_path path, by ignoring the pth file at the suffix
        save_dir = args.eval_model_path[:args.eval_model_path.rfind('/')]
        print(f"save_dir: {save_dir}")
        eval_trained_epsilon = args.eval_model_path.split('/')[-1]
        eval_trained_epsilon = re.search('[\d][\d]?', eval_trained_epsilon).group()
        assert type(eval_trained_epsilon) == str
        epsilons_list = list(range(args.eval_epsilon_max+1))
        
        acc_eval = True
        df = None

        # Check whether the file in {save_dir}/eval_accuracy_{eval_trained_epsilon}.csv already exist:
        if os.path.exists(f'{save_dir}/eval_accuracy_{eval_trained_epsilon}.csv'):
            print(f"File {save_dir}/eval_accuracy_{eval_trained_epsilon}.csv already exists. Appending to it")
            df = pd.read_csv(f'{save_dir}/eval_accuracy_{eval_trained_epsilon}.csv')
            # Verify whether the file contains the same epsilons as the current run (compare number of lines in file)
            if len(df) != args.eval_epsilon_max+1:
                epsilons_list = epsilons_list[len(df)+1:]
            else:
                acc_eval = False
        if args.eval_uncertainty:
            epsilons_list = list(range(args.eval_epsilon_max+1))
            uncertainty_dicts_list = []
        print(f"epsilons_list len: {len(epsilons_list)}")
        if acc_eval:
            eval_results = []
            train_results = []
        args.rc_curve_save_pth = f'{save_dir}/rc_curve_{eval_trained_epsilon}.pkl'
        for epsilon in epsilons_list:
            test_acc, uncertainty_dict = adv_eval(model, test_loader, args, epsilon/255, uncertainty_evaluation=args.eval_uncertainty)
            if acc_eval:
                train_results.append(adv_eval(model, train_loader, args, epsilon/255))
                eval_results.append(test_acc)
                print(f"Evaluated epsilon:{epsilon} , Test Accuracy: {eval_results[-1]*100}% , Train Accuracy: {train_results[-1]*100}%")
            if args.eval_uncertainty:
                uncertainty_dicts_list.append(uncertainty_dict)

        if acc_eval:
            df = pd.DataFrame(list(zip(epsilons_list, eval_results, train_results)), columns=['epsilon', 'eval_results', 'train_results'])

        if args.eval_uncertainty:
            uncertainty_df = pd.DataFrame.from_dict(uncertainty_dicts_list)
            df = pd.concat([df, uncertainty_df], axis=1)
        df.to_csv(f'{save_dir}/eval_accuracy_{eval_trained_epsilon}.csv', index=False)
    
    # Finished running training or evaluation.
    exit