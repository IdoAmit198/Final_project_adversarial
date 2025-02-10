import argparse
import json
import os

from tqdm import tqdm
from utils.args import get_args
from utils.models import wide_resnet
from utils.models.preact_resnet import PreActResNet18 
import torch
import torchvision
from utils.data import load_dataloaders
from adv_train import adv_training, adv_eval
from torch.amp import GradScaler

import pandas as pd
import re

import numpy as np
import random

import wandb
from datetime import datetime
import pytz

Models = ['WideResNet28_10', 'WideResNet34_10', 'WideResNet34_20', 'resnet18', 'preact_resnet18', 'resnet50']
Datasets = ['cifar100', 'cifar10', 'flowers102', 'mnist', 'imagenet100', 'imagenet']

def Inference_Args(args):
    # Attempt to load the json in the args.eval_model_path
    eval_model_path = args.eval_model_path
    eval_model_path_dir = args.eval_model_path[:args.eval_model_path.rfind('/')]
    eval_uncertainty_flag = args.eval_uncertainty
    device = args.device
    pgd_step_size_factor = args.pgd_step_size_factor
    try:
        with open(f'{eval_model_path_dir}/args.json', 'r') as f:
            loaded_args = json.load(f)
        eval_model_path = args.eval_model_path
        args = loaded_args
        args['eval_model_path'] = eval_model_path

    except FileNotFoundError:
        print(f"File {eval_model_path_dir}/args.json not found. Will do our best with the model path")
        file_dirs = args.eval_model_path.split('/')
        model_name = [model for model in Models if model in args.eval_model_path][0]
        # Extract the dataset
        dataset_name = [dataset for dataset in Datasets if dataset in args.eval_model_path]
        if len(dataset_name) == 0:
            dataset_name = 'cifar10'
        else:
            dataset_name = dataset_name[0]
        # Extract the pgs num steps
        pgd_steps_dir = [dir for dir in file_dirs if 'pgd_steps_' in dir]
        if len(pgd_steps_dir) == 0:
            pgd_steps = 10
        else:
            pgd_steps = int(pgd_steps_dir[0].split('pgd_steps_')[-1])
        # Extract the seed
        seed_dir = [dir for dir in file_dirs if 'seed_' in dir]
        if len(seed_dir) == 0:
            seed = -1
        else:
            seed = int(seed_dir[0].split('seed_')[-1])
        # Extract the optimizer
        optimizer_dir = [dir for dir in file_dirs if 'optimizer_' in dir]
        if len(optimizer_dir) == 0:
            optimizer = 'SGD'
        else:
            optimizer = optimizer_dir[0].split('optimizer_')[-1]
        ATAS_dir = [dir for dir in file_dirs if 'ATAS_' in dir]
        if len(ATAS_dir) == 0:
            ATAS = False
        else:
            ATAS = ATAS_dir[0].split('ATAS_')[-1]
        eval_trained_epsilon = re.search(r'[\d][\d]?', file_dirs[-1]).group()
        target_agnostic_dir = [dir for dir in file_dirs if 'agnostic_loss_' in dir][0]
        whether_agnostic = 'True' if 'True' in target_agnostic_dir else 'False'
        train_method = None
        if 'adaptive' in args.eval_model_path:
            train_method = 'adaptive'
        elif 're_introduce' in args.eval_model_path:
            train_method = 're_introduce'
        else:
            train_method = 'vanilla'
        new_args = {
            'dataset': dataset_name,
            'model_name': model_name,
            'optimizer': optimizer,
            'pgd_steps': pgd_steps,
            'seed': seed,
            'eval_trained_epsilon': eval_trained_epsilon,
            'whether_agnostic': whether_agnostic,
            'train_method': train_method,
            'Inference': True,
            'eval_model_path': args.eval_model_path,
            'ATAS': ATAS,
            'eval_model_path': eval_model_path,
            'fine_tune': None
        }
        args = new_args
        args['log_name'] = f"{args['model_name']}_train method_{args['train_method']}_agnostic_loss_{args['whether_agnostic']}_seed_{args['seed']}_max epsilon_{int(args['eval_trained_epsilon'])}"
    args = argparse.Namespace(**args)
    args.eval_uncertainty = eval_uncertainty_flag
    args.eval_epsilon_max = 32
    args.device = device
    args.pgd_step_size_factor = pgd_step_size_factor
    return args
        


if __name__ == '__main__':
    print("Started")

    args = get_args(description='Adversarial training')
    # args.Train = True
    # args.fine_tune = 'clean'
    # args.GradAlign = True
    # adjust pgd_steps_size according to a paper:
    # GradAlign https://arxiv.org/pdf/2007.02617
    if args.pgd_num_steps == 1:
        args.pgd_step_size_factor = 1.0
    elif args.pgd_num_steps == 2:
        args.pgd_step_size_factor = 0.5
    else:
        args.pgd_step_size_factor = 0.2
    
    # Adjust learning rate in ase of fine-tuning
    if args.fine_tune:
        args.learning_rate /= 10
        
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.max_epsilon = args.max_epsilon/255
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")
    args.cpu_num = len(os.sched_getaffinity(0))
    print(f"args:\n{args}")
    
    train_loader, validation_loader, test_loader = load_dataloaders(args)
    args.num_classes=10 # Cifar10
    if args.dataset in ['cifar100', 'imagenet100']:
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
    if 'wide' in args.model_name.lower():
        model = getattr(wide_resnet, args.model_name)(num_classes=args.num_classes)
    elif 'preact' in args.model_name.lower():
        args.atas_c = 0.01
        model = PreActResNet18(num_classes=args.num_classes)
    else:
        weights = None
        if args.fine_tune == 'clean':
            weights = "IMAGENET1K_V1"
        elif args.fine_tune == 'adversarial':
            pass
        model = torchvision.models.get_model(args.model_name,num_classes=args.num_classes, weights=weights)
    
    timezone = pytz.timezone('Asia/Jerusalem')
    # args.Train = True
    if args.Train:
        model = model.to(args.device)
        # model = torch.compile(model)
        model.forward = torch.compile(model.forward)
        if args.optimizer == 'Adam':
            # TODO: Refactor later to better practice.
            args.learning_rate = 1e-3
        # wanbd logging initialization
        args.log_name = f'{args.model_name}_train method_{args.train_method}_agnostic_loss_{args.agnostic_loss}_seed_{args.seed}_max epsilon_{int(args.max_epsilon*255)}'
        args.date_stamp = datetime.now(timezone).strftime("%d/%m_%H:%M")
        run = wandb.init(project="Adversarial-adaptive-project", name=f'{args.date_stamp}_{args.log_name}', entity = "ido-shani-proj", config=args)
        wandb.define_metric("step")
        wandb.define_metric("Epoch")
        wandb.define_metric("Train epochs loss", step_metric="Epoch")
        wandb.define_metric("Train epochs accuracy", step_metric="Epoch")
        wandb.define_metric("Train Clean epochs accuracy", step_metric="Epoch")
        wandb.define_metric("Test epochs accuracy", step_metric="Epoch")
        wandb.define_metric("Test epochs loss", step_metric="Epoch")
        wandb.define_metric("Train epochs clean loss", step_metric="Epoch")
        wandb.define_metric("Train epochs targeted loss", step_metric="Epoch")
        wandb.define_metric("Validation epochs accuracy", step_metric="Epoch")
        wandb.define_metric("Validation best accuracy", step_metric="Epoch")
        wandb.define_metric("Epsilons_metrics/min_epsilon", step_metric="Epoch")
        wandb.define_metric("Epsilons_metrics/max_epsilon", step_metric="Epoch")
        wandb.define_metric("Epsilons_metrics/mean_epsilon", step_metric="Epoch")
        wandb.define_metric("Epsilons_metrics/re_introduce_cur_prob", step_metric="Epoch")
        wandb.define_metric("train_lr", step_metric="Epoch")
        # Define the save_dir and save the args in that dir as a json file.
        additional_folder = 'sanity_check/' if args.sanity_check else ''
        save_dir = f"saved_models/{args.dataset}/fine_tune_{args.fine_tune}/{args.model_name}/{additional_folder}seed_{args.seed}/train_method_{args.train_method}/agnostic_loss_{args.agnostic_loss}/optimizer_{args.optimizer}/pgd_steps_{args.pgd_num_steps}"
        if os.path.exists(save_dir) and args.sanity_check:
            print(f"Sanity check model already exists in {save_dir}. Will train another one and save it in a different folder.")
            additional_folder = 'sanity_check_2-new/'
            save_dir = f"saved_models/{args.model_name}/{additional_folder}seed_{args.seed}/train_method_{args.train_method}/agnostic_loss_{args.agnostic_loss}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        args.save_dir = save_dir
        args.eval_model_path = f"{args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth"
        with open(f'{save_dir}/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(f"args saved in {save_dir}/args.json")
        # Initialize scaler for amp
        args.scaler = GradScaler()
        # Actual training
        adv_training(model, train_loader, validation_loader, test_loader, args)
    
    if args.Inference:
        if args.Train:
            args.eval_model_path = f"{args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth"
        else:
            # args.log_name = f'{args.model_name}_train method_{args.train_method}_agnostic_loss_{args.agnostic_loss}_seed_{args.seed}_max epsilon_{int(args.max_epsilon*255)}'
            # args.date_stamp = datetime.now(timezone).strftime("%d/%m_%H:%M")
            args = Inference_Args(args)
            run = wandb.init(project="Adversarial-adaptive-project", name=f"Inference_{args.log_name}", entity = "ido-shani-proj", config=args)
            # wandb.init(project="Adversarial-adaptive-project", name=f'{args.date_stamp}_{args.log_name}', entity = "ido-shani-proj", config=args)
        wandb.define_metric("Epsilon")
        wandb.define_metric("Inference/ Test", step_metric="Epsilon")
        wandb.define_metric("Inference/ Train", step_metric="Epsilon")
        if args.eval_uncertainty:
            # AUROC,AURC,ece15,confidence_mean
            wandb.define_metric("Inference/ ECE15", step_metric="Epsilon")
            wandb.define_metric("Inference/ AUROC", step_metric="Epsilon")
            wandb.define_metric("Inference/ AURC", step_metric="Epsilon")
            wandb.define_metric("Inference/ Confidence Mean", step_metric="Epsilon")

        # load statce_dict of the trained model from given path
        print(args)
        print(f"eval_model_path is {args.eval_model_path}")
        if os.path.isfile(args.eval_model_path):
            model.load_state_dict(torch.load(args.eval_model_path))
        else:
            raise FileNotFoundError(f"Model file not found: {args.eval_model_path}")
        # model.load_state_dict(torch.load(args.eval_model_path))
        print("Model loaded")
        model = model.to(args.device)
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
            if len(df) <= args.eval_epsilon_max+1:
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
        # Initialize scaler for amp
        args.scaler = GradScaler()
        time = datetime.now(timezone).strftime("%d/%m %H:%M - ")
        for epsilon in tqdm(epsilons_list, desc=f'{time}Eval'):
            test_acc, uncertainty_dict = adv_eval(model, test_loader, args, epsilon/255, uncertainty_evaluation=args.eval_uncertainty)
            if acc_eval:
                train_acc, _ = adv_eval(model, train_loader, args, epsilon/255, uncertainty_evaluation=False)
                train_results.append(train_acc)
                eval_results.append(test_acc)
                print(f"Evaluated epsilon:{epsilon} , Test Accuracy: {eval_results[-1]*100}% , Train Accuracy: {train_results[-1]*100}%")
            if args.eval_uncertainty:
                uncertainty_dicts_list.append(uncertainty_dict)
            time = datetime.now(timezone).strftime("%d/%m %H:%M - ")

        if acc_eval:
            df = pd.DataFrame(list(zip(epsilons_list, eval_results, train_results)), columns=['epsilon', 'eval_results', 'train_results'])

        if args.eval_uncertainty:
            uncertainty_df = pd.DataFrame.from_dict(uncertainty_dicts_list)
            df = pd.concat([df, uncertainty_df], axis=1)
        df.to_csv(f'{save_dir}/eval_accuracy_{eval_trained_epsilon}.csv', index=False)
        for idx, row in df.iterrows():
            epsilon = row['epsilon']
            eval = row['eval_results']
            train = row['train_results']
            wandb_log_metrics = {
                'Inference/ Test': eval*100,
                'Inference/ Train': train*100,
                'Epsilon': epsilon
            }
            if args.eval_uncertainty:
                wandb_log_metrics.update({
                    'Inference/ ECE15': row['ece15'],
                    'Inference/ AUROC': row['AUROC'],
                    'Inference/ AURC': row['AURC'],
                    'Inference/ Confidence Mean': row['confidence_mean']
                })
            wandb.log(wandb_log_metrics)
        # wandb.finish()
        run.finish()
    
    # Finished running training or evaluation.
    exit
