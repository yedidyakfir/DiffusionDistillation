from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

class CelebaFolderDataset(Dataset):
    def __init__(self, path: str, transform, resolution: int = 256):
        self.path = Path(path)

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(list(self.path.iterdir()))

    def __getitem__(self, index):
        full_path = self.path / f"{index:06d}.jpg"
        img = Image.open(str(full_path))
        img = img.convert("RGB")  # Convert image to RGB mode
        img = transforms.ToTensor()(img)  # Correct way to convert PIL image to tensor
        img = self.transform(img)  # Apply additional transformations if needed
        return img, 1
