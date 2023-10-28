from torch.utils.data import Dataset
import os
import numpy as np


class FlattenedDoodleSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get all .npy files from the directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.class_to_idx = {f.rsplit('.', 1)[0]: idx for idx, f in enumerate(self.image_files)}

        # Flatten the dataset
        self.images = []
        self.labels = []
        for idx, image_file in enumerate(self.image_files):
            class_images = np.load(os.path.join(data_dir, image_file))
            self.images.extend(class_images)
            self.labels.extend([idx] * len(class_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, label
