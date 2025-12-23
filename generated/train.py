import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import COVID19Dataset
from model import model

# Hyperparameters
learning_rate = 0.001  # TODO: Specify learning rate
num_epochs = 160

# Load dataset
train_dataset = COVID19Dataset()  # TODO: Split dataset into train/val/test
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)  # TODO: Use weighted loss
        loss.backward()
        optimizer.step()
    # TODO: Add validation and evaluation metrics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')