import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet, SegNet
from dataset import LungCTDataset

# TODO: Load dataset and split into training, validation, and testing sets

def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # TODO: Implement training loop
        for images, labels in train_loader:
            # TODO: Forward pass, compute loss, backward pass, and optimizer step
            pass

if __name__ == '__main__':
    # TODO: Initialize model, criterion, optimizer, and dataloaders
    model = UNet()  # or SegNet()
    criterion = torch.nn.CrossEntropyLoss()  # TODO: Use weighted loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # TODO: Adjust learning rate
    train_loader = DataLoader(LungCTDataset(...), batch_size=2, shuffle=True)  # TODO: Fill in dataset details
    train(model, train_loader, criterion, optimizer, num_epochs=160)