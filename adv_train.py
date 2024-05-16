from tqdm import tqdm
import torch
from torch.nn import functional as F
import wandb
import os

def PGD(model, x, y, epsilons, pgd_num_steps, targeted=False):
    model.eval()
    x_pert = x.clone().detach().requires_grad_(True)
    x_pert = x_pert + (torch.zeros_like(x_pert).uniform_(-1,1)*epsilons)
    for i in range(pgd_num_steps):
        # with torch.cuda.amp.autocast():
        y_score = model(x_pert)
        loss = F.cross_entropy(y_score, y)
        # loss.backward()
        grad = torch.autograd.grad(loss.mean(), [x_pert])[0].detach()
        x_grad = torch.sign(grad)
        pgd_step_size = epsilons/5
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

# @torch.compile
def adv_training(model, train_loader, test_loader, args):
    #Initialize scheduler and optimizer
    # scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 180], gamma=0.1)
    #Actual training+eval loop
    re_introduce_prob_step_size = 2*args.re_introduce_prob/args.max_epochs
    re_introduce_cur_prob = 0
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
            # epsilons = epsilons.to(args.device).requires_grad_(False)
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
            x_pert = PGD(model, x, y, epsilons_to_pgd, args.pgd_num_steps)
                # epsilons = epsilons.to('cpu')
            #else:
            #    x_pert = PGD(model, x, y, args.max_epsilon, args.pgd_num_steps)
            if args.agnostic_loss:
                # Generate y_targeted with random different labels than y
                y_targeted = torch.randint(0, 9, (x.shape[0],), device=args.device)
                y_targeted[y_targeted>=y] += 1
                x_targeted_pert = PGD(model, x, y_targeted, epsilons_to_pgd, args.pgd_num_steps, targeted=True)
            model.train()
            # with torch.cuda.amp.autocast():
            y_score = model(x_pert)
            loss_pert = F.cross_entropy(y_score, y)
            epoch_loss_pert += loss_pert.item()
            if args.agnostic_loss:
                y_score_targeted = model(x_targeted_pert)
                loss_targeted = F.cross_entropy(y_score_targeted, y)
                y_clean_score = model(x)
                loss_clean = F.cross_entropy(y_clean_score, y)
                total_loss = (loss_pert + loss_clean + loss_targeted)/3
                loss_targeted_epoch += loss_targeted.item()
                loss_clean_epoch += loss_clean.item()
            else:
                total_loss = loss_pert
            total_loss_epoch += total_loss.item()
            # x_pert.to('cpu')
            y_pred = torch.argmax(y_score, dim=1)
            incorrect = (y_pred!=y).to('cpu')
            train_error_samples += incorrect.sum().item()
            if args.train_method == 'adaptive' or args.train_method == 're_introduce':
                # incorrect = y_pred!=y
                tmp = args.epsilon_step_size*incorrect
                epsilons -= tmp
                train_loader.dataset.epsilons[indices] = epsilons
                train_loader.dataset.epsilons[indices] = epsilon_clamp(train_loader.dataset.epsilons[indices], args.max_epsilon)
            # wandb.log({"Train steps loss": loss.item()})
            
            optimizer.zero_grad()
            total_loss.backward()
            # loss.backward()
            # scaler.step(optimizer)
            optimizer.step()
            # scaler.update()
            # Empty cache
            del x, x_pert, y
            torch.cuda.empty_cache()
        ## Calculate accuracy
        train_accuracy = 1 - train_error_samples/train_samples_counter
        print(f"Epoch {epoch+1}: Train total Loss: {total_loss_epoch/len(train_loader)}, Train Accuracy: {train_accuracy*100}%")

        # Eval epoch for each 5 epochs
        epsilons_list = train_loader.dataset.epsilons
        min_epsilon = epsilons_list.min().item()
        max_epsilon = epsilons_list.max().item()
        mean_epsilon = epsilons_list.mean().item()

        if epoch%5==0:
            model.eval()
            test_error_samples = 0
            test_samples_counter = 0
            for batch in tqdm(test_loader, desc=f'Eval epoch {epoch+1}'):
                indices, epsilons, x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                test_samples_counter += x.shape[0]
                x_pert = PGD(model, x, y, args.max_epsilon, args.pgd_num_steps)
                # with torch.cuda.amp.autocast():
                y_score = model(x_pert)
                x_pert.to('cpu')
                y_pred = torch.argmax(y_score, dim=1)
                incorrect = y_pred!=y
                test_error_samples += incorrect.sum().item()
                del x, x_pert, y
                torch.cuda.empty_cache()

            test_accuracy = 1 - test_error_samples/test_samples_counter
            print(f"Epoch {epoch+1}: Test Accuracy: {test_accuracy*100}%")
            wandb.log({
                    # "Train epochs loss": total_loss/len(train_loader),
                #    "Train epochs clean loss": loss_clean_epoch/len(train_loader) if args.agnostic_loss else 0,    
                #    "Train epochs accuracy": train_accuracy*100,
                #    "train_lr": scheduler.get_last_lr()[0],
                    "Test epochs accuracy": test_accuracy*100,
                    # "Epsilons_metrics/min_epsilon": min_epsilon,
                    # "Epsilons_metrics/max_epsilon": max_epsilon,
                    # "Epsilons_metrics/mean_epsilon": mean_epsilon,
                    # "Epsilons_metrics/re_introduce_cur_prob": re_introduce_cur_prob, 
                       "Epoch": epoch+1})
        if args.agnostic_loss:
            wandb.log({"Train epochs targeted loss": loss_targeted_epoch/len(train_loader),
                        "Train epochs clean loss": loss_clean_epoch/len(train_loader),
                        "Train epochs pert loss": epoch_loss_pert/len(train_loader), 
                    "Epoch": epoch+1})
        if args.train_method == 're_introduce':
            wandb.log({"Epsilons_metrics/re_introduce_cur_prob": re_introduce_cur_prob,
                        "Epoch": epoch+1})
        wandb.log({"Train epochs loss": total_loss/len(train_loader),
                   "Train epochs accuracy": train_accuracy*100,
                   "train_lr": scheduler.get_last_lr()[0],
                   "Epsilons_metrics/min_epsilon": min_epsilon,
                    "Epsilons_metrics/max_epsilon": max_epsilon,
                    "Epsilons_metrics/mean_epsilon": mean_epsilon,
                   "Epoch":epoch+1})
        scheduler.step()
        

    # Save the trained model at given path and verify whether the directory exists
    save_dir = f"saved_models/{args.model_name}/seed_{args.seed}/train_method_{args.train_method}/agnostic_loss_{args.agnostic_loss}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), f"{save_dir}/max_epsilon_{int(args.max_epsilon*255)}.pth")
    # Load the saved model
    # model.load_state_dict(torch.load(f"saved_models/{args.model_name}/seed_{args.seed}/train_method_{args.train_method}/max_epsilon_{int(args.max_epsilon*255)}.pth"))
        # wandb.log({})
        # print(f"Accuracy: {accuracy*100}%")


                # x_pert = PGD(model, x_pert, y_batch, args.max_epsilon)
        