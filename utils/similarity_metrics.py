import os
import torch
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid(dataloader_1, dataloader_2, device='cuda'):
    """
    Calculate the Frechet Inception Distance (FID) between two sets of images.
    Args:
        real_images (torch.Tensor): A batch of real images.
        fake_images (torch.Tensor): A batch of generated images.
        device (str): Device to perform calculations on ('cuda' or 'cpu').
    Returns:
        float: The FID score.
    """
    
    fid_meter = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # update “real” then “fake” (labels are arbitrary since both are just sets)
    for imgs in dataloader_1:
        fid_meter.update(imgs.to(device), real=True)

    for imgs in dataloader_2:
        fid_meter.update(imgs.to(device), real=False)

    fid = fid_meter.compute().item()
    # print(f"TorchMetrics FID: {fid:.4f}")
    return fid


def calculate_cosine(dataloader_1, dataloader_2, device='cuda'):
    """
    Calculate the cosine similarity between two sets of images.
    Args:
        dataloader_1 (DataLoader): DataLoader for the first set of images.
        dataloader_2 (DataLoader): DataLoader for the second set of images.
        device (str): Device to perform calculations on ('cuda' or 'cpu').
    Returns:
        float: The cosine similarity score.
    """
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    all_cosine_scores = []

    for imgs_1, imgs_2 in zip(dataloader_1, dataloader_2):
        imgs_1 = imgs_1.to(device)
        imgs_2 = imgs_2.to(device)

        # Flatten the images to compute cosine similarity
        imgs_1_flat = imgs_1.view(imgs_1.size(0), -1)
        imgs_2_flat = imgs_2.view(imgs_2.size(0), -1)

        scores = cosine_similarity(imgs_1_flat, imgs_2_flat).cpu()
        # Append the scores to the list as single tensors
        for i in range(scores.size(0)):
            all_cosine_scores.append(scores[i])
        #all_cosine_scores.append(scores.cpu())

    # return torch.cat(all_cosine_scores).mean().item()
    return sum(all_cosine_scores)/len(all_cosine_scores)

def calculate_MSE(dataloader_1, dataloader_2, device='cuda'):
    """
    Calculate the Mean Squared Error (MSE) between two sets of images.
    Args:
        dataloader_1 (DataLoader): DataLoader for the first set of images.
        dataloader_2 (DataLoader): DataLoader for the second set of images.
        device (str): Device to perform calculations on ('cuda' or 'cpu').
    Returns:
        float: The MSE score.
    """
    mse_loss = torch.nn.MSELoss()

    all_mse_scores = []

    for imgs_1, imgs_2 in zip(dataloader_1, dataloader_2):
        imgs_1 = imgs_1.to(device)
        imgs_2 = imgs_2.to(device)

        scores = mse_loss(imgs_1, imgs_2).cpu()
        # for i in range(scores.size(0)):
        #     all_mse_scores.append(scores[i])
        all_mse_scores.append(scores.item())

    return sum(all_mse_scores)/len(all_mse_scores)