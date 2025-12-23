import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COVID19Dataset
from model import UNet

# TODO: Load dataset and split into train, validation, test
train_dataset = COVID19Dataset(image_paths=[], masks_paths=[], transform=None)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = UNet()
criterion = nn.CrossEntropyLoss()  # TODO: Implement weighted cross-entropy
optimizer = optim.Adam(model.parameters())

for epoch in range(160):
    model.train()
    for images, masks in train_loader:
        # TODO: Implement training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    # TODO: Implement validation and logging
