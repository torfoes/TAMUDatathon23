import os
import numpy as np
from torch.utils.data import Dataset


class DoodleSet(Dataset):
    def __init__(self, data_dir, transform=None, limit=None):
        """
        Args:
            data_dir (str): Directory with all the images in .npy format.
            transform (callable, optional): Optional transform to be applied on an image.
            limit (int, optional): Limit the number of images loaded into the dataset.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Get all .npy files from the directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Limit the number of files if specified
        if limit:
            self.image_files = self.image_files[:limit]

        # Create class-to-index mapping
        classes = set([f.rsplit('.', 1)[0] for f in self.image_files])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])

        # Load the array of images for the class and extract a specific image
        all_images = np.load(img_path)
        image = all_images[0]  # Here, we're taking the first image as an example
        class_name = self.image_files[idx].rsplit('.', 1)[0]
        label = self.class_to_idx[class_name]

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, label