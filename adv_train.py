from tqdm import tqdm
import torch
from torch.nn import functional as F
import wandb

def PGD(model, x, y, epsilons, pgd_num_steps, scaler=None):
    model.eval()
    x_pert = x.clone().detach().requires_grad_(True)
    x_pert = x_pert + (torch.zeros_like(x_pert).uniform_(-1,1)*epsilons)
    for i in range(pgd_num_steps):
        # print(f'iteration: {i} in pgd')
        with torch.cuda.amp.autocast():
            y_score = model(x_pert)
        loss = F.cross_entropy(y_score, y)
        # loss.backward()
        grad = torch.autograd.grad(scaler.scale(loss).mean(), [x_pert])[0].detach()
        x_grad = grad.sign()
        pgd_step_size = epsilons/5
        pgd_grad_step = pgd_step_size*x_grad
        assert pgd_grad_step.shape == x.shape
        # print(f"pgd_grad_step.shape: {pgd_grad_step.shape}")
        x_pert = x_pert + pgd_grad_step
        x_pert = torch.max(torch.min(x_pert, x+epsilons), x-epsilons)
        x_pert = torch.clamp(x_pert, 0, 1)
    return x_pert

def epsilon_clamp(epsilons, max_epsilon):
    return torch.clamp(epsilons, 0, max_epsilon)

# @torch.compile
def adv_training(model, train_loader, args):
    #Initialize scheduler and optimizer
    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140, 180], gamma=0.1)
    #Actual training loop
    for epoch in range(args.max_epochs):
        epoch_loss = 0
        samples_counter = 0
        error_samples = 0
        for batch in tqdm(train_loader):
            indices, epsilons, x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            samples_counter += x.shape[0]
            if args.train_method == 'adaptive':
                epsilons += args.epsilon_step_size
                x_pert = PGD(model, x, y, epsilons, args.pgd_num_steps, scaler)
            else:
                x_pert = PGD(model, x, y, args.max_epsilon, args.pgd_num_steps, scaler)
            # print("Passed PGD")
            model.train()
            with torch.cuda.amp.autocast():
                y_score = model(x_pert)
            y_pred = torch.argmax(y_score, dim=1)
            incorrect = y_pred!=y
            error_samples += incorrect.sum().item()
            if args.train_method == 'adaptive':
                incorrect = y_pred!=y
                train_loader.dataset.epsilons[indices] -= args.epsilon_step_size*incorrect
                train_loader.dataset.epsilons[indices] = epsilon_clamp(train_loader.dataset.epsilons[indices], args.max_epsilon)
            loss = F.cross_entropy(y_score, y)
            wandb.log({"Train steps loss": loss.item()})
            epoch_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()
        scheduler.step()
        # print(f"Epoch {epoch} completed")
        ## Calculate accuracy
        accuracy = 1 - error_samples/samples_counter
        print(f"Epoch {epoch}: Loss: {loss.item()}, Accuracy: {accuracy*100}%")
        wandb.log({"Train epochs loss": epoch_loss/len(train_loader)})
        wandb.log({"Train epochs accuracy": accuracy})
        # print(f"Accuracy: {accuracy*100}%")


                # x_pert = PGD(model, x_pert, y_batch, args.max_epsilon)
        