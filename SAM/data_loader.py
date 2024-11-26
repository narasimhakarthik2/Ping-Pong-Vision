import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os


class TableDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        # Resize for consistency
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask.astype(np.float32), (256, 256))  # Ensure mask is a numeric type

        # Normalize image
        image = image / 255.0
        mask = mask.astype(np.float32)

        # Convert to tensors
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def get_dataloader(image_dir, mask_dir, batch_size=8, shuffle=True):
    dataset = TableDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
