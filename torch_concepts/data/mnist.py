import numpy as np
import random
import torch

from torchvision.datasets import MNIST

def _colorize(image: torch.Tensor, color: str) -> torch.Tensor:
    # Create an image with 3 channels (RGB)
    colored_image = torch.zeros(3, 28, 28)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    return colored_image


class ColorMNISTDataset(MNIST):
    """
    The color MNIST dataset is a modified version of the MNIST dataset where
    each digit is colored either red or green. The concept labels are the digit
    and the color of the digit. The task is to predict whether the digit is
    even or odd.

    Attributes:
        root: The root directory where the dataset is stored.
        train: Whether to load the training or test split. Default is False.
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels.
            Default is None.
        download: Whether to download the dataset if it does not exist. Default
            is False.
        random: Whether to colorize the digits randomly. Default is True.
    """
    def __init__(
        self,
        root: str,
        train: bool = False,
        transform = None,
        target_transform = None,
        download: bool = False,
        random: bool = True,
    ):
        super(ColorMNISTDataset, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.random = random
        self.concept_attr_names = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            'red',
            'green',
        ]
        self.task_attr_names = ['even', 'odd']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, digit = self.data[index], int(self.targets[index])

        # Colorize the image
        if self.random:
            color = 'red' if random.random() < 0.5 else 'green'
        else:
            color = 'red' if digit <= 5 else 'green'
        # Remove channel dimension of the grayscale image
        colored_image = _colorize(image.squeeze(), color)

        # Create the concept label
        concept_label = np.zeros(12)  # 10 digits + 2 colors
        concept_label[digit] = 1
        concept_label[10] = 1 if color == 'red' else 0
        concept_label[11] = 1 if color == 'green' else 0

        # Create the target label
        target_label = 1 if digit % 2 == 0 else 0
        target_label = [target_label, 1 - target_label]

        return (
            colored_image,
            torch.tensor(concept_label, dtype=torch.float32),
            torch.tensor(target_label, dtype=torch.float32),
        )
