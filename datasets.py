from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CelebaFolderDataset(Dataset):
    """
    Custom Dataset class for Celeba images stored in a folder.
    
    Args:
        path (str): Path to the dataset directory.
        transform (callable): Transformations to be applied on each image.
        resolution (int): Resolution of the images.
    """
    
    def __init__(self, path: str, transform, resolution: int = 256):
        self.path = Path(path)  # Convert path to Path object for easier handling
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Number of images.
        """
        return len(list(self.path.iterdir()))  # Count the number of files in the directory

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding label (dummy label in this case).
        
        Args:
            index (int): Index of the image to retrieve.
        
        Returns:
            tuple: Transformed image and a dummy label (1).
        """
        full_path = self.path / f"{index:06d}.jpg"  # Construct the file path
        img = Image.open(str(full_path))  # Open the image
        img = img.convert("RGB")  # Convert image to RGB mode
        img = transforms.ToTensor()(img)  # Convert PIL image to tensor
        img = self.transform(img)  # Apply additional transformations if needed
        return img, 1  # Return image and dummy label
