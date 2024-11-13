
import torch

from torchvision.datasets import CelebA
from typing import List


class CelebADataset(CelebA):
    """
    The CelebA dataset is a large-scale face attributes dataset with more than
    200K celebrity images, each with 40 attribute annotations. This class
    extends the CelebA dataset to extract concept and task attributes based on
    class attributes.

    The dataset can be downloaded from the official
    website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Default is 'train'.
        transform: The transformations to apply to the images. Default is None.
        download: Whether to download the dataset if it does not exist. Default
            is False.
        class_attributes: The class attributes to use for the task. Default is
            None.
    """
    def __init__(
        self, root: str, split: str = 'train',
        transform = None,
        download: bool = False,
        class_attributes: List[str] = None,
    ):
        super(CelebADataset, self).__init__(
            root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        # Set the class attributes
        if class_attributes is None:
            # Default to 'Attractive' if no class_attributes provided
            self.class_idx = [self.attr_names.index('Attractive')]
        else:
            # Use the provided class attributes
            self.class_idx = [
                self.attr_names.index(attr) for attr in class_attributes
            ]

        self.attr_names = [string for string in self.attr_names if string]

        # Determine concept and task attribute names based on class attributes
        self.concept_attr_names = [
            attr for i, attr in enumerate(self.attr_names)
            if i not in self.class_idx
        ]
        self.task_attr_names = [self.attr_names[i] for i in self.class_idx]

    def __getitem__(self, index: int):
        image, attributes = super(CelebADataset, self).__getitem__(index)

        # Extract the target (y) based on the class index
        y = torch.stack([attributes[i] for i in self.class_idx])

        # Extract concept attributes, excluding the class attributes
        concept_attributes = torch.stack([
            attributes[i] for i in range(len(attributes))
            if i not in self.class_idx
        ])

        return image, concept_attributes, y
