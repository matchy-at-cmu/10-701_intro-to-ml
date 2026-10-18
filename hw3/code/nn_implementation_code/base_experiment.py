from collections import Counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from custom_modules import Linear, CrossEntropyLoss, Sigmoid


class FashionMNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize linear and sigmoid layers
        pass


    def forward(self, x):
        # TODO: Implement forward pass 
        pass


@torch.no_grad()
def evaluate(model, loader, loss_func, device):
    """
    Evaluate the model over the full dataset.
    Returns (loss, accuracy) averaged over total samples, not batches.
    """
    # TODO: Implement evaluation function
    ...

    return loss, accuracy


def train(model, loader, optimizer, loss_func, device):
    """Train the model for one epoch. Mutates model in place, returns nothing."""
    # TODO: Implement training loop for one epoch
    ...
    pass



if __name__ == "__main__":
    trainset = torchvision.datasets.FashionMNIST(root='./', train=True,
                                                 download=True, transform=transforms.ToTensor())

    testset = torchvision.datasets.FashionMNIST(root='./', train=False,
                                                download=True, transform=transforms.ToTensor())

    weights = torch.load("weights.pt")
    print(weights)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    with torch.no_grad():
        ...

    for epoch in tqdm(range(50)):
        train(...)
        loss, acc = evaluate(...)
        ...
    ...
