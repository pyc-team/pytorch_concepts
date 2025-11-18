import numpy as np
import random
import torch
from typing import Tuple

from torchvision.datasets import MNIST
from torchvision.transforms import transforms


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


class MNISTAddition(MNIST):
    """
        The MNIST addition dataset is a modified version of the MNIST dataset
        where each image is a concatenation of two MNIST images and the target
        label is the sum of the two digits. The concept label is a one-hot
        encoding of the two digits.

        Attributes:
            concept_names: The names of the concept labels.
            task_names: The names of the task labels.
            root: The root directory where the dataset is stored.
            train: Whether to load the training or test split. Default is False.
            transform: The transformations to apply to the images. Default is
                None.
            target_transform: The transformations to apply to the target labels.
                Default is None.
            download: Whether to download the dataset if it does not exist.
                Default is False.
    """
    name = "mnist_addition"
    n_concepts = 20
    n_tasks = 19
    concept_names = [
        "0_left", "1_left", "2_left", "3_left", "4_left",
        "5_left", "6_left", "7_left", "8_left", "9_left",
        "0_right", "1_right", "2_right", "3_right", "4_right",
        "5_right", "6_right", "7_right", "8_right", "9_right",
    ]
    task_names = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16", "17", "18",
    ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_shape = (1, 28, 56)
    input_dim = 28 * 56

    def __init__(self, root, train,
                 target_transform=None, download=True):
        super(MNISTAddition, self).__init__(
            root,
            train,
            self.transform,
            target_transform,
            download,
        )

    def __getitem__(
        self,
        index,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the first image and target
        img_1, target_1 = super(MNISTAddition, self).__getitem__(index)

        # Get a second image and target. To get a different image, we need to
        # sample a different index. To do this, we pick the index from the end
        img_2, target_2 = super(MNISTAddition, self).__getitem__(-index)

        # Horizontally concat the two images and sum targets to get task label
        img = torch.cat((img_1, img_2), dim=2)

        # Sum the targets to get the task label
        y = target_1 + target_2

        # One hot encoding of the concept label on 20 digits
        c = torch.zeros(20)
        c[target_1] = 1
        c[target_2 + 10] = 1

        return img, c, y


class PartialMNISTAddition(MNISTAddition):
    """
    The partial MNIST addition dataset is a modified version of the MNIST
    addition dataset where the concept annotation is partial. The concept
    associated with the second digit is not provided.
    """
    name = "partial_mnist_addition"
    n_concepts = 10
    concept_names = [
        "0_left", "1_left", "2_left", "3_left", "4_left",
        "5_left", "6_left", "7_left", "8_left", "9_left",
    ]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, c, y = super(PartialMNISTAddition, self).__getitem__(index)
        c = c[:10]
        return x, c, y


class MNISTEvenOdd(MNIST):
    """
    The MNIST even-odd dataset is a modified version of the MNIST dataset where
    the task is to predict whether the digit is even or odd. The concept label
    is a one-hot encoding of the digit.

    Attributes:
        concept_names: The names of the concept labels.
        task_names: The names of the task labels.
        root: The root directory where the dataset is stored.
        train: Whether to load the training or test split. Default is False.
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels.
            Default is None.
        download: Whether to download the dataset if it does not exist. Default
            is False.
    """
    name = "mnist_even_odd"
    n_concepts = 10
    n_tasks = 2
    concept_names = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    task_names = ["odd", "even"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_shape = (1, 28, 28)
    input_dim = 28 * 28

    def __init__(self, root, train,
                 target_transform=None, download=True):
        super(MNISTEvenOdd, self).__init__(root, train, self.transform,
                                           target_transform, download)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = super(MNISTEvenOdd, self).__getitem__(index)

        # One hot encoding of the concept label on 10 digits
        c = torch.zeros(10)
        c[y] = 1

        # Task label is 1 if digit is even, 0 if digit is odd
        t = 1 if y % 2 == 0 else 0

        return x, c, t
