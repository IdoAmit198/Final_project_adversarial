from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset

import torch
import numpy as np
import random

class DatasetWithMeta(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.epsilons = torch.zeros(len(dataset), dtype=torch.float32)

    def __getitem__(self, index):
        x,y = self.dataset.__getitem__(index)
        epsilon = self.epsilons[index]
        return(index, epsilon, x,y)
    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_dataloaders(batch_size:int = 32, seed:int = 42):
    g = torch.Generator()
    g.manual_seed(seed)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    train_ds = DatasetWithMeta(train_dataset)
    val_ds = DatasetWithMeta(val_dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(val_ds, batch_size=int(batch_size/2), shuffle=True, num_workers=24, pin_memory=True)
    return train_loader, test_loader