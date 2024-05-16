from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset

import torch

class DatasetWithMeta(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        # self.targets = targets
        self.epsilons = torch.zeros(len(dataset), dtype=torch.float32)

    def __getitem__(self, index):
        x,y = self.dataset.__getitem__(index)
        epsilon = self.epsilons[index]
        return(index, epsilon, x,y)
        # return (self.inputs[index], self.targets[index], self.metas[index])
    def __len__(self):
        return len(self.dataset)
    
def load_dataloaders(batch_size=32):
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    test_loader = DataLoader(val_ds, batch_size=int(batch_size/2), shuffle=True, num_workers=24, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader