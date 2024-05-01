from pathlib import Path

import cv2
from torch.utils.data import Dataset


class CelebaFolderDataset(Dataset):
    def __init__(self, path: str, transform, resolution: int = 256):
        self.path = Path(path)

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(list(self.path.iterdir()))

    def __getitem__(self, index):
        full_path = self.path / f"{index:06d}.jpg"
        img = cv2.imread(str(full_path))
        img = self.transform(img)
        return img, 1
