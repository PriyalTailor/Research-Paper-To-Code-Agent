import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms as transforms

class LungCTDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: Load image and label from NIFTI files
        image = nib.load(self.image_paths[idx]).get_fdata()
        label = nib.load(self.label_paths[idx]).get_fdata()

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# TODO: Add data augmentation techniques if needed