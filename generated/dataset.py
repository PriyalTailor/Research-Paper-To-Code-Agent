import torch
from torchvision import datasets, transforms

class COVID19Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        # TODO: Load dataset images and masks
        self.transform = transform

    def __len__(self):
        # TODO: Return the total number of images
        return 0

    def __getitem__(self, idx):
        # TODO: Load and return a single image and its mask
        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = COVID19Dataset(transform=transform)
