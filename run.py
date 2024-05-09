from utils.args import get_args
import torch
from torch import nn
import torchvision
from utils.data import load_dataloaders
from adv_train import adv_training
import warnings


if __name__ == '__main__':
    print("Started")
    # torch._dynamo.config.suppress_errors = True
    # torch.set_float32_matmul_precision('high')

    args = get_args(description='Adversarial training')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if args.device=='cuda':
    #     gpu_ok = False
    #     device_cap = torch.cuda.get_device_capability()
    #     if device_cap in ((7, 0), (8, 0), (9, 0)):
    #         gpu_ok = True
    #         print(f"gpu ok: {gpu_ok}")
    #     if not gpu_ok:
    #         print(
    #         "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.")
    print(args.device)
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model_name, weights=None)
    num_classes=10
    # if args.data =='cifar10':
    #     num_classes=10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(args.device)
    # model = torch.compile(model)
    # input = torch.ones(1,3,32,32)
    train_loader, test_loader = load_dataloaders(args.batch_size)

    scaler = torch.cuda.amp.GradScaler()
    adv_training(model, train_loader, args)


    # print(lx.shape[0])
    # print(model(input).shape)
    # for i in range(2):
    #     for batch in test_loader:
    #         index,epsilon, x, y= batch
    #         print(f"x.shape:{x.shape}, y.shape:{y.shape}\nindex:{index}\nepsilon:{epsilon}")
    #         test_loader.dataset.epsilons[index] = 8/255
    #         print('\n\n')
    #         print(model(x).shape)
    #         break
    # print(model)
    exit
    # run_adv_training(args)