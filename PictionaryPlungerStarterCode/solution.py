# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

# Solution.py
import torch
import numpy as np
from torchvision.transforms import transforms
import torch.nn as nn
import cv2


# CNN model definition (assuming a simple CNN architecture)
class CNN(nn.Module):
    def __init__(self, num_classes=345):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Helper function to convert strokes to image format
def strokes_to_image(strokes):
    image = np.zeros((256, 256), dtype=np.uint8)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, 3)
    return image


class Solution:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])

        self.num_classes = 345  # Assuming there are 345 classes, modify as needed
        self.model = CNN(num_classes=self.num_classes)

        # Assuming the model weights are loaded somewhere, if needed

        self.model.eval()

    # This is a signal that a new drawing is about to be sent
    def new_case(self):
        pass

    # Given a stroke, return a string of your guess
    def guess(self, x: list[int], y: list[int]) -> str:
        strokes = [x, y]
        image = strokes_to_image(strokes)
        image_tensor = self.transform(image).unsqueeze(0)
        outputs = self.model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        # Return the label name based on the prediction
        # Here, you'd map the predicted index to its corresponding label name
        return self.label_to_name[predicted.item()]

    # This function is called when you get
    def add_score(self, score: int):
        print(score)
