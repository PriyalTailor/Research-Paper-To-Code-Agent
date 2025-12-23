import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # TODO: Define U-Net architecture layers

    def forward(self, x):
        # TODO: Implement forward pass
        return x

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # TODO: Define SegNet architecture layers

    def forward(self, x):
        # TODO: Implement forward pass
        return x

# Choose one of the models
model = UNet()  # or SegNet()

