from tqdm import tqdm
import torch
from torch.nn import functional as F
import wandb
# import os
# from torch.cuda.amp import GradScaler, autocast

from torchvision.transforms import v2
from utils.data import Cutout 

import pytz
from datetime import datetime

from utils.uncertainty_metrics import log_uncertainty
from utils.warm_schduler import WarmupCosineLR

from autoattack import AutoAttack
import csv

def PGD(model, x, y, epsilons, pgd_num_steps, args, targeted=False, gdnorms = None):
    # If epsilons are all zeros, no need to run PGD, simply return x.
    if isinstance(epsilons, torch.Tensor):
        if not torch.any(epsilons):
            return x
    elif isinstance(epsilons, (int, float)):
        if epsilons == 0:
            return x
    model.eval()
    # Freeze model for the PGD attack
    for p in model.parameters():
        p.requires_grad = False
    x_pert = x.clone().detach().requires_grad_(True)
    x_pert = x_pert + (torch.zeros_like(x_pert).uniform_(-1,1)*epsilons)
    for i in range(pgd_num_steps):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_score = model(x_pert)
            loss = F.cross_entropy(y_score, y)
        grad = torch.autograd.grad(loss.mean(), [x_pert])[0].detach()
        x_grad = torch.sign(grad)
        if gdnorms is not None:
            with torch.no_grad():
                cur_gdnorm = torch.norm(grad.view(len(x_pert), -1), dim=1).detach() ** 2 * (1 - args.atas_beta) + gdnorms * args.atas_beta
                step_sizes = 1 / (1 + torch.sqrt(cur_gdnorm) / args.atas_c) * 2 * 8 / 255
                step_sizes = torch.clamp(step_sizes, args.atas_min_step_size, args.atas_max_step_size)
            pgd_step_size = step_sizes.view(-1, 1, 1, 1).expand_as(grad)
            gdnorms = cur_gdnorm
        else:
            pgd_step_size = epsilons*args.pgd_step_size_factor
        pgd_grad_step = pgd_step_size*x_grad
        assert pgd_grad_step.shape == x.shape
        if not targeted:
            x_pert = x_pert + pgd_grad_step
        else:
            x_pert = x_pert - pgd_grad_step
        x_pert = torch.max(torch.min(x_pert, x+epsilons), x-epsilons)
        x_pert = torch.clamp(x_pert, 0, 1)
        # detach x_pert to avoid increasing of computation graph, and require grad since `detach` removes the grad requirement.
        x_pert = x_pert.detach().requires_grad_(True)
        
    # Unfreeze model after the PGD attack
    for p in model.parameters():
        p.requires_grad = True
    if gdnorms is not None:
        return x_pert, gdnorms
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
        if len(batch) == 4:
            _, _, x, y = batch
        else:
            x, y = batch
        x, y = x.to(args.device), y.to(args.device)
        test_samples_counter += x.shape[0]
        x_pert = PGD(model, x, y, evaluated_epsilon, 100, args)
        # Freeze model for evaluation after PGD unfreezed it.
        # for p in model.parameters():
        #     p.requires_grad = False
        # Model isn't training so if we don't train a PGD attack, no need for gradients.
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_score = model(x_pert)
        # y_score = args.scaler.scale(y_score)
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
    return test_accuracy, None

def configure_optimizers(model, train_dataloader, args):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=torch.tensor(args.learning_rate),
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=torch.tensor(args.learning_rate),
            weight_decay=args.weight_decay,
        )

    steps_per_epoch = len(train_dataloader.batch_sampler)
    total_steps = args.max_epochs * steps_per_epoch
    if args.scheduler == "WarmupCosineLR":
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * args.warmup_ratio, max_epochs=total_steps)
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=args.learning_rate/10)
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[total_steps*0.4, total_steps*0.7, total_steps*0.9], gamma=0.1)
    elif args.scheduler == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.3, step_size_up=total_steps / 2, step_size_down=total_steps / 2)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} is not implemented.")
    return optimizer, scheduler

def adv_training(model, train_loader, validation_loader, test_loader, args):
    # For date and time logs
    timezone = pytz.timezone('Asia/Jerusalem')
    # Initialize the optimizer and the scheduler
    optimizer, scheduler = configure_optimizers(model, train_loader, args)
    
    @torch.compile(fullgraph=False)
    def step():
        """
        compiled optimized step function.
        Apply both the optimizer and scheduler step.
        """
        optimizer.step()
        scheduler.step()
    # if args.ATAS:
    gdnorm_list = torch.zeros(len(train_loader.dataset), device=args.device)
    # Initialize augmentation transformation for clean samples:
    # clean_transform = v2.Compose([
    #     # v2.RandomHorizontalFlip(),
    #     v2.RandomRotation(15),
    # ])
    if 'wide' in args.model_name.lower() or 'preact' in args.model_name.lower():
        image_size = 32
    else:
        image_size = 224    
    # clean_transform.transforms.append(Cutout(n_holes=1, length=image_size//2))
    val_best_accuracy = 0
    for epoch in range(args.max_epochs):
        time = datetime.now(timezone).strftime("%d/%m %H:%M - ")
        
        epoch_loss_pert = 0
        loss_targeted_epoch = 0
        loss_clean_epoch = 0
        total_loss_epoch = 0
        train_samples_counter = 0
        train_error_samples = 0
        clean_error_samples = 0
        # Training epoch
        grad_alignment = []
        for batch in tqdm(train_loader, desc=f'{time}Training epoch {epoch+1}'):
            indices, epsilons, x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            train_samples_counter += x.shape[0]
            epsilons_to_pgd = args.max_epsilon
            if args.train_method in ['adaptive', 're_introduce']:
                epsilons += args.epsilon_step_size
                epsilons_tensor = epsilons.clone().to(args.device).requires_grad_(False)
                if args.train_method == 're_introduce':
                    # mask_matrix = torch.zeros([epsilons_tensor.shape[0], args.max_epsilon], device=args.device)
                    # In each row i in mask_matrix, put 1 in all the columns up to the entry epsilons_tensor[i]
                    # For example, if epsilons_tensor[i] = 3, then mask_matrix[i, :3] = 1
                    cols = torch.arange(0, 255*args.max_epsilon+1, device=args.device)
                    # Compare and generate the mask
                    mask_matrix = (cols <= 255*epsilons_tensor.unsqueeze(1)).int()
                    # Create a matrix where each row represent the geometric distribution and each entry has the probability for that given number to be picked according to a geomrtric distribution.
                    # Make it tensorised
                    p = 0.1  # Adjust the probability of success as needed,  the paramter for geometric distribution
                    # Compute the geometric probabilities for all values in the range [1, 64]
                    cols = torch.arange(0, 255*args.max_epsilon+1)  # 1-indexed range
                    geometric_probabilities = p * (1 - p) ** (cols - 1)  # Probability for each column

                    # Expand geometric probabilities to match the rows of the mask matrix
                    probability_matrix = geometric_probabilities.unsqueeze(0).repeat(epsilons_tensor.shape[0], 1).to(args.device)
                    # print(probability_matrix)
                    probability_matrix *= mask_matrix

                    # Normalize each row to ensure the probabilities sum to 1
                    row_sums = probability_matrix.sum(dim=1, keepdim=True)
                    normalized_probability_matrix = probability_matrix/row_sums
                    
                    # Sample from the normalized probability matrix,
                    # Add 1 to avoid using no pertubatino,
                    # And divide by 255 to get the epsilon value.
                    samples = (torch.multinomial(normalized_probability_matrix, num_samples=1))/255
                    # epsilons_tensor -= samples.squeeze(1)
                    epsilons_tensor = samples.squeeze(1)

                    # Legacy code
                    # # Sample a random boolean tensor of the len of epsilon where probability to get 1 is 'p'
                    # # p = 0.5
                    # random_bool = torch.rand(len(epsilons_tensor), device=args.device) < re_introduce_cur_prob
                    # # Sample a random uniform tensor of the len of epsilons
                    # random_uniform = torch.rand(len(epsilons_tensor), device=args.device)
                    # random_uniform *= epsilons_tensor*random_bool
                    # epsilons_tensor -= random_uniform
                epsilons_tensor = epsilons_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                epsilons_to_pgd = epsilons_tensor
            
            gdnorm_batch = gdnorm_list[indices] if args.ATAS else None
            x_pert = PGD(model, x, y, epsilons_to_pgd, args.pgd_num_steps, args, gdnorms=gdnorm_batch)
            if args.ATAS:
                # If we use ATAS, PGD returns two objects, the perturbed image and the gradient norm
                x_pert, new_gdnorm_batch = x_pert
                gdnorm_list[indices] = new_gdnorm_batch
            if args.agnostic_loss:
                # Generate y_targeted with random different labels than y
                y_targeted = torch.randint(0, args.num_classes - 1, (x.shape[0],), device=args.device)
                y_targeted[y_targeted>=y] += 1
                x_targeted_pert = PGD(model, x, y_targeted, epsilons_to_pgd, args.pgd_num_steps, args, targeted=True, gdnorms=gdnorm_batch)
                if args.ATAS:
                    # If we use ATAS, PGD returns two objects, the perturbed image and the gradient norm
                    x_targeted_pert, _ = x_targeted_pert
            model.train()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_score = model(x_pert)
                loss_pert = F.cross_entropy(y_score, y)
            # x_augmented = clean_transform(x) if args.augment else x
            x_augmented = x
            if args.GradAlign:
                x_augmented.requires_grad = True
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_clean_score = model(x_augmented)
                loss_clean = F.cross_entropy(y_clean_score, y)
                a=1
            # if args.use_clean_loss:
            # scaled_loss_clean = args.scaler.scale(loss_clean)
            # scaled_loss_pert = args.scaler.scale(loss_pert)
            epoch_loss_pert += loss_pert.item()
            if args.agnostic_loss:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    y_score_targeted = model(x_targeted_pert)
                    loss_targeted = F.cross_entropy(y_score_targeted, y)
                # scaled_loss_targeted = args.scaler.scale(loss_targeted)
                if args.use_clean_loss:
                    total_loss = (loss_pert + loss_clean + loss_targeted)/3
                    total_loss_epoch += (loss_pert + loss_clean + loss_targeted).item()/3
                else:
                    total_loss = (loss_pert + loss_targeted)/2
                    total_loss_epoch += (loss_pert + loss_targeted).item()/2
                loss_targeted_epoch += loss_targeted.item()
            else:
                if args.use_clean_loss:
                    total_loss = (loss_pert + loss_clean)/2
                    total_loss_epoch += (loss_pert + loss_clean).item()/2
                else:
                    total_loss = loss_pert
                    total_loss_epoch += loss_pert.item()
            # If we use GradAlign, make another computation, and add it to the total loss term.
            if args.GradAlign:
                # Sample a perturbation from a uniform compute the gradients w.r.t the perturbed sample
                x_clone = x.clone().detach().requires_grad_(True)
                x_clone = x_clone + (torch.zeros_like(x_clone).uniform_(-1,1)*epsilons_to_pgd)
                # compute grad w.r.t the uniform perturbed samples
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    y_unif_score = model(x_clone)
                    unif_pert_loss = F.cross_entropy(y_unif_score, y)
                # scaled_unif_pert_loss = args.scaler.scale(unif_pert_loss)
                # scaled_clean_loss = args.scaler.scale(clean_loss)
                # Compute the gradients w.r.t the input sample
                grad_unif_pert = torch.autograd.grad(unif_pert_loss.mean(), [x_clone])[0].detach()
                # Utilize the already computed clean loss from above
                grad_clean = torch.autograd.grad(loss_clean.mean(), [x_augmented])[0].detach()
                grad1 = grad_unif_pert.reshape(len(grad_unif_pert), -1)
                grad2 =  grad_clean.reshape(len(grad_clean), -1)
                cos_sim = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
                grad_alignment.append(cos_sim.mean().item())
                reg = args.grad_align_lambda * (1.0 - cos_sim.mean())
                total_loss = total_loss + reg
            clean_error_samples += (torch.argmax(y_clean_score, dim=1) != y).sum().item()
            loss_clean_epoch += loss_clean.item()
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
            # optimizer.step()
            # args.scaler.step(optimizer)
            # scheduler.step()
            step()
            # log learing rate
            # wandb.log({"train_lr": scheduler.get_last_lr()[0]})
            # args.scaler.update()
            # Empty cache
            # del x, x_pert, y
            # torch.cuda.empty_cache()
            time = datetime.now(timezone).strftime("%d/%m %H:%M - ")
        ## Calculate accuracy
        train_accuracy = 1 - train_error_samples/train_samples_counter
        # if args.use_clean_loss:
        train_clean_accuracy = 1 - clean_error_samples/train_samples_counter
        print(f"Epoch {epoch+1}: Train total Loss: {total_loss_epoch/len(train_loader)}, Train Accuracy: {train_accuracy*100}%")
        epsilons_list = train_loader.dataset.epsilons
        min_epsilon = epsilons_list.min().item()
        max_epsilon = epsilons_list.max().item()
        mean_epsilon = epsilons_list.mean().item()

        # Validation epoch each epoch to evaluate the model and save the best model.
        model.eval()
        validation_samples_counter = 0
        time = datetime.now(timezone).strftime("%d/%m %H:%M - ")
        # validation_epsilons = [0, 8/255, 16/255, 32/255]
        validation_epsilons = [0, args.max_epsilon/4, args.max_epsilon/2, args.max_epsilon]
        epsilons_error_samples = [0] * len(validation_epsilons)

        for batch in tqdm(validation_loader, desc=f'{time}Validation epoch {epoch+1}'):
            _, _, val_x, val_y = batch
            val_x, val_y = val_x.to(args.device), val_y.to(args.device)
            validation_samples_counter += val_x.shape[0]
            for idx, val_epsilon in enumerate(validation_epsilons):
                val_x_pert = PGD(model, val_x, val_y, val_epsilon, 20, args)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    y_score = model(val_x_pert)
                epsilons_error_samples[idx] += (y_score.argmax(dim=1) != val_y).sum().item()

        validation_accuracy_list = [1 - epsilon_error_sample/validation_samples_counter for epsilon_error_sample in epsilons_error_samples]
        
        # validation_epsilons_weights = [0.1, 0.04305, 0.01853, 0.00343]
        # normalized_validation_epsilons_weight = [weight/sum(validation_epsilons_weights) for weight in validation_epsilons_weights]
        # validation_accuracy = sum([normalized_weight * epsilon_accuracy for normalized_weight, epsilon_accuracy in zip(normalized_validation_epsilons_weight, validation_accuracy_list)])
        clean_val_acc = validation_accuracy_list[0]
        validation_accuracy = sum(validation_accuracy_list[1:])/len(validation_accuracy_list[1:])
        if validation_accuracy > val_best_accuracy and clean_val_acc > args.dataset_clean_min_threshold:
            val_best_accuracy = validation_accuracy
            print(f"New best model found with validation accuracy of {val_best_accuracy*100}%!")
            print("Saving model...")
            torch.save(model.state_dict(), f"{args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
            print(f"Model saved at {args.save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
        print(f"Epoch {epoch+1}: Validation Accuracy: {validation_accuracy*100}%, Best Validation Accuracy: {val_best_accuracy*100}%")

        # Eval epoch for each 5 epochs
        if epoch%5==0:
            test_error_samples = 0
            test_samples_counter = 0
            test_loss_pert = 0
            time = datetime.now(timezone).strftime("%d/%m %H:%M - ")
            for batch in tqdm(test_loader, desc=f'{time}Eval epoch {epoch+1}'):
                _, _, x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                test_samples_counter += x.shape[0]
                x_pert = PGD(model, x, y, args.max_epsilon, 20, args)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    y_score = model(x_pert)
                    test_loss_pert += F.cross_entropy(y_score, y).item()
                # test_loss_pert = args.scaler.scale(test_loss_pert).item()
                x_pert.to('cpu')
                y_pred = torch.argmax(y_score, dim=1)
                incorrect = y_pred!=y
                test_error_samples += incorrect.sum().item()
                del x, x_pert, y

            test_accuracy = 1 - test_error_samples/test_samples_counter
            print(f"Epoch {epoch+1}: Test Accuracy: {test_accuracy*100}%, Test Loss: {test_loss_pert/len(test_loader)}")
            wandb.log({
                        "Test epochs accuracy": test_accuracy*100,
                        "Test epochs loss": test_loss_pert/len(test_loader),
                       "Epoch": epoch+1})

        wandb_log_dict = {  "Train epochs loss": epoch_loss_pert/len(train_loader),
                            # "Train epochs clean loss": loss_clean_epoch/len(train_loader),
                            "Train epochs accuracy": train_accuracy*100,
                            # "Train Clean epochs accuracy": train_clean_accuracy*100,
                            "Validation/Mean Validation accuracy": validation_accuracy*100,
                            "Validation/Clean Accuracy": validation_accuracy_list[0]*100,
                            f"Validation/Epsilon {validation_epsilons[1]*255}": validation_accuracy_list[1]*100,
                            f"Validation/Epsilon {validation_epsilons[2]*255}": validation_accuracy_list[2]*100,
                            f"Validation/Epsilon {validation_epsilons[3]*255}": validation_accuracy_list[3]*100,
                            "Validation/Validation best accuracy": val_best_accuracy*100,
                            "train_lr": scheduler.get_last_lr()[0],
                            "Epsilons_metrics/min_epsilon": min_epsilon,
                            "Epsilons_metrics/max_epsilon": max_epsilon,
                            "Epsilons_metrics/mean_epsilon": mean_epsilon,
                            "Epoch":epoch+1
        }
        wandb_log_dict["Train epochs clean loss"] = loss_clean_epoch/len(train_loader)
        wandb_log_dict["Train Clean epochs accuracy"] = train_clean_accuracy*100
        if args.agnostic_loss:
            wandb_log_dict['Train epochs targeted loss'] = loss_targeted_epoch/len(train_loader)
        if args.ATAS:
            wandb_log_dict['ATAS/Gradients-Norm mean'] = gdnorm_list.mean().item()
            logging_step_sizes = 1 / (1 + torch.sqrt(gdnorm_list) / args.atas_c) * 2 * 8 / 255
            logging_step_sizes = torch.clamp(logging_step_sizes, args.atas_min_step_size, args.atas_max_step_size)
            wandb_log_dict['ATAS/ Step-Size mean'] = logging_step_sizes.mean().item()
        if args.GradAlign:
            wandb_log_dict['GradAlign/Gradients Alignment'] = sum(grad_alignment)/len(grad_alignment)
        # Actual logging
        wandb.log(wandb_log_dict)
        


def evaluate_model_autoattack(model, dataloader, max_eps:int, csv_filename:str, device='cuda', attacks_names_list:list[str] = None):
    """
    Evaluates the L∞ robust accuracy of a model using AutoAttack over epsilons 1 to max_eps.
    Also computes the clean accuracy of the model.
    
    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation dataset.
        max_eps (int): Maximum epsilon value (evaluated from 1 to max_eps; eps is scaled by 1/255).
        device (str): Device to run evaluation on.
        csv_filename (str): File name to save CSV results.
        
    Returns:
        results (dict): Dictionary with keys 'clean_accuracy' and each epsilon mapping to robust accuracy.
    """
    model.eval()
    model.to(device)
    results = {eps: {} for eps in range(max_eps + 1)}
    
    # Compute clean accuracy over the dataloader
    clean_correct = 0
    total = 0
    for batch in dataloader:
        if len(batch) == 4:
            _, _, images, labels = batch
        else:
            images, labels = batch
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        clean_correct += (preds == labels).sum().item()
        total += labels.size(0)
    clean_acc = clean_correct / total
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")

    # List of attacks to evaluate
    attacks = ["apgd-ce"] if attacks_names_list is None else attacks_names_list

    # log clean accuracy
    for attack in attacks:
        results[0][attack] = clean_acc
        wandb.log({
            f"AA_Inference/{attack}": clean_acc*100,
             "Epsilon":0})

    # Evaluate each attack individually
    for attack in attacks:
        
        # Loop over epsilon values from 1 to max_eps
        for eps in range(1, max_eps + 1):
            eps_scaled = eps / 255.0  # Scale epsilon for images in [0,1]
            print(f"\nEvaluating epsilon {eps} (scaled: {eps_scaled:.5f})")
            robust_correct = 0
            total_batch = 0

            # Create an AutoAttack instance with only the chosen attack.
            adversary = AutoAttack(model, norm='Linf', eps=eps_scaled, version='custom', 
                                   attacks_to_run=[attack], device=device, verbose=False)
            # Set parameters for each attack:
            if attack == "apgd-ce":
                # Untargeted APGD-CE with no restarts
                adversary.apgd.n_restarts = 1
            elif attack == "apgd-dlr":
                # Targeted APGD-DLR: set to 9 target classes, no restarts.
                adversary.apgd_targeted.n_restarts = 1
                adversary.apgd_targeted.n_target_classes = 9

            robust_correct_attack = 0
            total_attack = 0
            
            # Process the dataset batch by batch
            progress_bar = tqdm(dataloader, desc=f"Epsilon {eps}, Attack {attack}")
            for batch in progress_bar:
                if len(batch) == 4:
                    _, _, images, labels = batch
                else:
                    images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                # Run the attack on the batch.
                # Note: run_standard_evaluation returns adversarial examples only for images that were
                # initially correctly classified.
                # print(f"Before running batch {batch_idx}")
                _ , non_robust_total_num = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])
                # delta = (x_adv - images).abs().view(images.size(0), -1).max(1)[0]
                # print("max pixel delta:", delta)
                # with torch.no_grad():
                #     preds = model(x_adv).argmax(dim=1)
                robust_correct_attack += images.shape[0] - non_robust_total_num
                total_attack += labels.size(0)
                progress_bar.set_postfix(current_acc=f"{(robust_correct_attack/total_attack):.4f}")
            
            robust_acc_attack = (robust_correct_attack / total_attack)
            results[eps][attack] = robust_acc_attack
            print(f"Attack {attack} at epsilon {eps}: Robust Accuracy = {robust_acc_attack * 100:.2f}%")
            wandb.log({
            f"AA_Inference/{attack}": robust_acc_attack*100,
             "Epsilon":eps})
    
    # --- Save results to a CSV file ---
    # CSV columns: Attack name, Epsilons values starting from 0 to max_eps.
    # Each row corresponds to the robust accuracy of the attack at the given epsilon.
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epsilon"] + attacks)
        for eps in range(max_eps + 1):
            csvwriter.writerow([eps] + [results[eps][attack] for attack in attacks])
    
    print("\nFinal Results:")
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    for attack in attacks:
        print(f"Attack {attack}:")    
        for eps in range(1, max_eps + 1):
            print(f"Epsilon {eps}: {results[eps][attack] * 100:.2f}%")
    print(f"\nResults saved to {csv_filename}")
    
    return results

# Example usage:
# Assume you have already defined your model and dataloader for your validation set.
# results = evaluate_model_autoattack(model, dataloader, max_eps=32, device='cuda', csv_filename='results.csv')
