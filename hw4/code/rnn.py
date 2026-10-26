import math
import tqdm
import string
import time
import random
import numpy as np
import torch
from torch import nn, optim

from reference_data import get_dataloaders

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # TODO: Initialize embeddings, model layers, layer normalization, and activation function

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass including layer normalization
        pass
    
    def init_hidden(self) -> torch.Tensor:
        # TODO: Initialize hidden state to zeros
        pass


def train(data_loader, model, criterion, optimizer):
    test_avg_loss = 0
    num_correct = 0

    # TODO: Set model to training mode

    # TODO: Implement training loop
    # Hint: See reference_data.py for the format of data_loader batches

    # TODO: remove "pass" and uncomment the line below
    pass
    # for ... in tqdm.tqdm(data_loader, leave=False):
        # TODO: Convert labels to LongTensors and move to device

        # TODO: Initialize hidden state

        # TODO: Perform forward pass and calculate avg loss over all time steps

        # TODO: Calculate number of correct predictions

        # TODO: Backward pass, update weights, and zero gradients

    # Uncomment the line below once the training loop is implemented to print 
    # the loss and accuracy at the end of each epoch
    # print(f"Train loss: {test_avg_loss/len(data_loader.dataset)} | Accuracy: {num_correct/len(data_loader.dataset)}")
    pass


def test(data_loader, rnn, criterion):
    test_avg_loss = 0
    num_correct = 0

    # TODO: Set model to eval mode

    # TODO: Implement testing loop
    # Hint: Do we want to update gradients when testing?

        # Uncomment the line below once the training loop is implemented to print 
        # the loss and accuracy at the end of each epoch
        # print(f"Test loss: {test_avg_loss/len(data_loader.dataset)} | Accuracy: {num_correct/len(data_loader.dataset)}")


def run(num_epochs, train_dataloader, test_dataloader, rnn, criterion, optimizer):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # TODO: Perform one epoch of training and testing
        pass


def main(
    # Hyperparameters
    vocabulary_size = 20000,
    batch_size = 128,
    input_size = 64,
    hidden_size = 128,
    max_review_length = 50,
    lr = 1e-4,
    num_epochs = 10,
    num_classes = 2
):

    train_dataloader, test_dataloader, full_vocab = get_dataloaders(vocabulary_size, max_review_length, batch_size)

    # TODO: Initialize model

    # TODO: Initialize loss function and optimizer
    
    # TODO: Run training and testing

    return train_loss_list, test_loss_list

if __name__ == '__main__':
    main()
    