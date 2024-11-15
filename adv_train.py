from tqdm import tqdm
import torch
from torch.nn import functional as F
import wandb
import os

from torchvision.transforms import v2
from utils.data import Cutout 


from utils.uncertainty_metrics import log_uncertainty
from utils.warm_schduler import WarmupCosineLR

def PGD(model, x, y, epsilons, args, targeted=False):
    model.eval()
    x_pert = x.clone().detach().requires_grad_(True)
    x_pert = x_pert + (torch.zeros_like(x_pert).uniform_(-1,1)*epsilons)
    for i in range(args.pgd_num_steps):
        y_score = model(x_pert)
        # print(f"x_pert shape: {x_pert.shape}")
        # print(f"y_score shape: {y_score.shape} , y shape: {y.shape}")
        loss = F.cross_entropy(y_score, y)
        grad = torch.autograd.grad(loss.mean(), [x_pert])[0].detach()
        x_grad = torch.sign(grad)
        pgd_step_size = epsilons*args.pgd_step_size_factor
        pgd_grad_step = pgd_step_size*x_grad
        assert pgd_grad_step.shape == x.shape
        if not targeted:
            x_pert = x_pert + pgd_grad_step
        else:
            x_pert = x_pert - pgd_grad_step
        x_pert = torch.max(torch.min(x_pert, x+epsilons), x-epsilons)
        x_pert = torch.clamp(x_pert, 0, 1)
    return x_pert

def epsilon_clamp(epsilons, max_epsilon):
    return torch.clamp(epsilons, 0, max_epsilon)

def adv_eval(model, test_loader, args, evaluated_epsilon, uncertainty_evaluation=False):
    model.eval()
    test_error_samples = 0
    test_samples_counter = 0
    samples_certainties = torch.empty((0, 2))
    for batch in test_loader:
        # Load only the samples, no need for the epsilons nor indices since we won't modify those.
        _, _, x, y = batch
        x, y = x.to(args.device), y.to(args.device)
        test_samples_counter += x.shape[0]
        x_pert = PGD(model, x, y, evaluated_epsilon, args)
        # Model isn't trainign so if we don't train a PGD attack, no need for gradients.
        with torch.no_grad():
            y_score = model(x_pert)
        y_pred = torch.argmax(y_score, dim=1)
        incorrect = y_pred!=y
        test_error_samples += incorrect.sum().item()
        if uncertainty_evaluation:
            # Only needed if we measure uncetainty estimation abilities of the models.
            probs = F.softmax(y_score, dim=1)
            confidence = probs.max(dim=1)[0].cpu()
            correctness = (probs.argmax(dim=1) == y).cpu()
            samples_certainties_batch = torch.stack([confidence.clone(), correctness.clone()])
            samples_certainties = torch.vstack((samples_certainties, samples_certainties_batch.transpose(0, 1))).cpu()

    test_accuracy = 1 - test_error_samples/test_samples_counter
    
    if uncertainty_evaluation:
        uncertainty_dict = log_uncertainty(samples_certainties, args.rc_curve_save_pth)
        return test_accuracy, uncertainty_dict
    return test_accuracy

def configure_optimizers(model, train_dataloader, args):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    steps_per_epoch = len(train_dataloader.batch_sampler)
    total_steps = args.max_epochs * steps_per_epoch
    if args.scheduler == "WarmupCosineLR":
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * args.warmup_ratio, max_epochs=total_steps)
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 180], gamma=0.1)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} is not implemented.")
    return optimizer, scheduler

def adv_training(model, train_loader, validation_loader, test_loader, args):
    # Initialize the optimizer and the scheduler
    optimizer, scheduler = configure_optimizers(model, train_loader, args)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 180], gamma=0.1)
    # Initialize a linear scheduler for the probability to re-introudce smaller epsilon values.
    # It rise steadily from 0 to args.re_introduce_prob in the first half of the training and then stays constant.
    re_introduce_prob_step_size = 2*args.re_introduce_prob/args.max_epochs
    re_introduce_cur_prob = 0
    # Actual training+eval loop. We make an evaluation every 5 epochs.
    
    # Initialize augmentation transformation for clean samples:
    clean_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
    ])
    if 'wide' in args.model_name.lower():
        image_size = 32
    else:
        image_size = 224    
    clean_transform.transforms.append(Cutout(n_holes=1, length=image_size//2))
    val_best_accuracy = 0
    for epoch in range(args.max_epochs):
        if re_introduce_cur_prob < args.re_introduce_prob:
            re_introduce_cur_prob += re_introduce_prob_step_size
        
        epoch_loss_pert = 0
        loss_targeted_epoch = 0
        loss_clean_epoch = 0
        total_loss_epoch = 0
        train_samples_counter = 0
        train_error_samples = 0
        # Training epoch
        for batch in tqdm(train_loader, desc=f'Training epoch {epoch+1}'):
            indices, epsilons, x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            train_samples_counter += x.shape[0]
            epsilons_to_pgd = args.max_epsilon
            if args.train_method == 'adaptive' or args.train_method == 're_introduce':
                epsilons += args.epsilon_step_size
                epsilons_tensor = epsilons.clone().to(args.device).requires_grad_(False)
                if args.train_method == 're_introduce':
                    # Sample a random boolean tensor of the len of epsilon where probability to get 1 is 'p'
                    # p = 0.5
                    random_bool = torch.rand(len(epsilons_tensor), device=args.device) < re_introduce_cur_prob
                    # Sample a random uniform tensor of the len of epsilons
                    random_uniform = torch.rand(len(epsilons_tensor), device=args.device)
                    random_uniform *= epsilons_tensor*random_bool
                    epsilons_tensor -= random_uniform
                epsilons_tensor = epsilons_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                epsilons_to_pgd = epsilons_tensor
            x_pert = PGD(model, x, y, epsilons_to_pgd, args)

            if args.agnostic_loss:
                # Generate y_targeted with random different labels than y
                y_targeted = torch.randint(0, 9, (x.shape[0],), device=args.device)
                y_targeted[y_targeted>=y] += 1
                x_targeted_pert = PGD(model, x, y_targeted, epsilons_to_pgd, args, targeted=True)
            model.train()
            y_score = model(x_pert)
            loss_pert = F.cross_entropy(y_score, y)
            epoch_loss_pert += loss_pert.item()
            x_augmented = clean_transform(x)
            y_clean_score = model(x_augmented)
            loss_clean = F.cross_entropy(y_clean_score, y)
            if args.agnostic_loss:
                y_score_targeted = model(x_targeted_pert)
                loss_targeted = F.cross_entropy(y_score_targeted, y)
                total_loss = (loss_pert + loss_clean + loss_targeted)/3
                loss_targeted_epoch += loss_targeted.item()
            else:
                total_loss = (loss_pert + loss_clean)/2
            loss_clean_epoch += loss_clean.item()
            total_loss_epoch += total_loss.item()
            y_pred = torch.argmax(y_score, dim=1)
            incorrect = (y_pred!=y).to('cpu')
            train_error_samples += incorrect.sum().item()
            if args.train_method == 'adaptive' or args.train_method == 're_introduce':
                tmp = args.epsilon_step_size*incorrect
                epsilons -= tmp
                train_loader.dataset.epsilons[indices] = epsilons
                train_loader.dataset.epsilons[indices] = epsilon_clamp(train_loader.dataset.epsilons[indices], args.max_epsilon)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # Empty cache
            del x, x_pert, y
            torch.cuda.empty_cache()
        ## Calculate accuracy
        train_accuracy = 1 - train_error_samples/train_samples_counter
        print(f"Epoch {epoch+1}: Train total Loss: {total_loss_epoch/len(train_loader)}, Train Accuracy: {train_accuracy*100}%")
        epsilons_list = train_loader.dataset.epsilons
        min_epsilon = epsilons_list.min().item()
        max_epsilon = epsilons_list.max().item()
        mean_epsilon = epsilons_list.mean().item()

        # Validation epoch each epoch to evaluate the model and save the best model.
        model.eval()
        validation_error_samples = 0
        validation_samples_counter = 0
        for batch in tqdm(validation_loader, desc=f'Validation epoch {epoch+1}'):
            indices, epsilons, val_x, val_y = batch
            val_x, val_y = x.to(args.device), y.to(args.device)
            validation_samples_counter += val_x.shape[0]
            val_x_pert = PGD(model, val_x, val_y, args.max_epsilon, args)
            y_score = model(val_x_pert)
            y_pred = torch.argmax(y_score, dim=1)
            incorrect = y_pred!=val_y
            validation_error_samples += incorrect.sum().item()
            del val_x, x_pert, val_y
            # torch.cuda.empty_cache()
        validation_accuracy = 1 - validation_error_samples/validation_samples_counter
        if validation_accuracy > val_best_accuracy:
            val_best_accuracy = validation_accuracy
            print(f"New best model found with validation accuracy of {val_best_accuracy*100}%!")
            print("Saving model...")
            torch.save(model.state_dict(), f"{args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
            print(f"Model saved at {args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
            # torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
        wandb.log({
                    "Validation epochs accuracy": validation_accuracy*100,
                    "Validation best accuracy": val_best_accuracy*100,
                    "Epoch": epoch+1})
        print(f"Epoch {epoch+1}: Validation Accuracy: {validation_accuracy*100}%, Best Validation Accuracy: {val_best_accuracy*100}%")

        # Eval epoch for each 5 epochs
        if epoch%5==0:
            test_error_samples = 0
            test_samples_counter = 0
            test_loss_pert = 0
            for batch in tqdm(test_loader, desc=f'Eval epoch {epoch+1}'):
                indices, epsilons, x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                test_samples_counter += x.shape[0]
                x_pert = PGD(model, x, y, args.max_epsilon, args)
                # with torch.cuda.amp.autocast():
                y_score = model(x_pert)
                test_loss_pert += F.cross_entropy(y_score, y).item()
                x_pert.to('cpu')
                y_pred = torch.argmax(y_score, dim=1)
                incorrect = y_pred!=y
                test_error_samples += incorrect.sum().item()
                del x, x_pert, y
                # torch.cuda.empty_cache()

            test_accuracy = 1 - test_error_samples/test_samples_counter
            print(f"Epoch {epoch+1}: Test Accuracy: {test_accuracy*100}%, Test Loss: {test_loss_pert/len(test_loader)}")
            wandb.log({
                        "Test epochs accuracy": test_accuracy*100,
                        "Test epochs loss": test_loss_pert/len(test_loader),
                       "Epoch": epoch+1})
        if args.agnostic_loss:
            wandb.log({"Train epochs targeted loss": loss_targeted_epoch/len(train_loader),
                        "Epoch": epoch+1})
        if args.train_method == 're_introduce':
            wandb.log({"Epsilons_metrics/re_introduce_cur_prob": re_introduce_cur_prob,
                        "Epoch": epoch+1})
        wandb.log({"Train epochs loss": epoch_loss_pert/len(train_loader),
                   "Train epochs clean loss": loss_clean_epoch/len(train_loader),
                   "Train epochs accuracy": train_accuracy*100,
                   "train_lr": scheduler.get_last_lr()[0],
                   "Epsilons_metrics/min_epsilon": min_epsilon,
                    "Epsilons_metrics/max_epsilon": max_epsilon,
                    "Epsilons_metrics/mean_epsilon": mean_epsilon,
                   "Epoch":epoch+1})
        

    # Save the trained model at given path and verify whether the directory exists
    # additional_folder = 'sanity_check/' if args.sanity_check else ''
    # save_dir = f"saved_models/{args.model_name}/{additional_folder}seed_{args.seed}/train_method_{args.train_method}/agnostic_loss_{args.agnostic_loss}"
    # if os.path.exists(save_dir) and args.sanity_check:
    #     print(f"Sanity check model already exists in {save_dir}. Will train another one and save it in a different folder.")
    #     additional_folder = 'sanity_check_2-new/'
    #     save_dir = f"saved_models/{args.model_name}/{additional_folder}seed_{args.seed}/train_method_{args.train_method}/agnostic_loss_{args.agnostic_loss}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(model.state_dict(), f"{args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
        