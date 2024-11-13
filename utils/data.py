from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.datasets import VisionDataset

import torch
import numpy as np
import random

# Implementing the Cutout augmentation
# Implementatino is taken from:
# https://discuss.pytorch.org/t/why-does-data-augmentation-decrease-validation-accuracy-pytorch-keras-comparison/29297/6
class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (B, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        b = img.size(0)
        c = img.size(1)
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((b, c, h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h, size=b)
            x = np.random.randint(w, size=b)

            y1 = (np.clip(y - self.length / 2, 0, h)).astype(int)
            y2 = (np.clip(y + self.length / 2, 0, h)).astype(int)
            x1 = (np.clip(x - self.length / 2, 0, w)).astype(int)
            x2 = (np.clip(x + self.length / 2, 0, w)).astype(int)

            for i in range(b):
                mask[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 0

        mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        img = img * mask.to(img.device)

        return img

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

def load_dataloaders(args, seed:int = 42):
    batch_size = args.batch_size
    g = torch.Generator().manual_seed(seed)
    # g.manual_seed(seed)
    image_size = 32
    if 'wide' in args.model_name.lower():
        image_size = 32
    else:
        image_size = 224    
    train_transform = v2.Compose([
        # v2.ToImage(),
        v2.Resize((image_size, image_size)),
        # v2.RandomHorizontalFlip(),
        # v2.RandomRotation(15),
        v2.ToTensor(),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])

    test_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])
        # train_transform = transforms.Compose([
        #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        #     transforms.Resize((image_size, image_size)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # test_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
    # train_transform.transforms.append(Cutout(n_holes=1, length=image_size//2))
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    # Split train_ds to train_ds and val_ds, with a ratio of 10% of the original train_ds size.
    train_ds, validation_ds = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=g)
    train_ds = DatasetWithMeta(train_ds)
    validation_ds = DatasetWithMeta(validation_ds)
    test_ds = DatasetWithMeta(test_dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.cpu_num, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size/2), shuffle=False, num_workers=args.cpu_num, pin_memory=True)
    validation_loader = DataLoader(validation_ds, batch_size=int(batch_size/2), shuffle=False, num_workers=args.cpu_num, pin_memory=True)
    return train_loader, validation_loader, test_loader