import os
import random
import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from tqdm import tqdm

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)

CONCEPT_NAMES = ['first_digit', 'second_digit']
TASK_NAMES = ['result']


def _generate_arithmetic_data(
    img_dir: str,
    mnist_root: str,
    train: bool,
    num_samples: int,
    img_size: int,
    seed: int,
    filename_offset: int = 0,
):
    """Generate MNIST arithmetic composite images and save to disk.

    Args:
        img_dir: Directory to save generated images.
        mnist_root: Root for MNIST download.
        train: Whether to use MNIST train split.
        num_samples: Number of samples to generate.
        img_size: Output image size (square).
        seed: Random seed.
        filename_offset: Starting index for filenames (to avoid collisions).

    Returns:
        Tuple of (filenames, concepts, tasks) lists.
    """
    random.seed(seed)
    np.random.seed(seed)

    mnist = datasets.MNIST(root=mnist_root, train=train, download=True, transform=None)

    resize_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
    ])

    try:
        font = ImageFont.truetype("arial.ttf", 200)
    except OSError:
        font = ImageFont.load_default()

    os.makedirs(img_dir, exist_ok=True)

    operators = ('+', '-', 'x', '/')

    operator_list = [random.choice(operators) for _ in range(num_samples)]

    filenames = []
    concepts_list = []
    tasks_list = []

    for idx in tqdm(range(num_samples), desc=f"Generating MNIST arithmetic ({'train' if train else 'test'})"):
        # Sample two digits, skip 0
        i1 = random.randint(0, len(mnist) - 1)
        i2 = random.randint(0, len(mnist) - 1)
        img1, a = mnist[i1]
        img2, b = mnist[i2]

        while a == 0 or b == 0:
            if a == 0:
                i1 = random.randint(0, len(mnist) - 1)
                img1, a = mnist[i1]
            if b == 0:
                i2 = random.randint(0, len(mnist) - 1)
                img2, b = mnist[i2]

        op = operator_list[idx]

        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == 'x':
            result = a * b
        elif op == '/':
            result = a / b
        else:
            raise ValueError(f"Unknown operator: {op}")

        # Create composite image
        canvas = Image.new("L", (84, 28), color=255)
        canvas.paste(img1, (0, 0))

        op_canvas = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(op_canvas)
        try:
            bbox = draw.textbbox((0, 0), op, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(op, font=font)
        x_pos = (28 - text_width) // 2
        y_pos = (28 - text_height) // 2
        draw.text((x_pos, y_pos), op, fill=255, font=font)

        canvas.paste(op_canvas, (28, 0))
        canvas.paste(img2, (56, 0))

        # Resize to final size
        final_img = resize_transform(canvas)

        fname = f"sample_{filename_offset + idx}.png"
        final_img.save(os.path.join(img_dir, fname))

        filenames.append(fname)
        concepts_list.append([float(a), float(b)])
        tasks_list.append([float(result)])

    return filenames, concepts_list, tasks_list


class MNISTArithmeticDataset(ConceptDataset):
    """MNIST Arithmetic dataset for regression with concept annotations.

    Composite images of two MNIST digits with an arithmetic operator between
    them. The concepts are the two digit values (treated as continuous).
    The regression task is the arithmetic result.

    Images in the training/validation splits are composed from MNIST train
    digits, while test images are composed from MNIST test digits, ensuring
    no digit-level leakage between train and test.

    Parameters
    ----------
    root : str, optional
        Root directory to store/load the dataset. Default: ``'./data/mnist_arithmetic'``.
    num_train_samples : int, optional
        Number of composite samples from MNIST train split (used for
        train + validation). Default: 10000.
    num_test_samples : int, optional
        Number of composite samples from MNIST test split. Default: 2000.
    val_size : float, optional
        Fraction of the train pool to use as validation. Default: 0.1.
    img_size : int, optional
        Output image size (square). Default: 224.
    seed : int, optional
        Random seed for reproducible generation. Default: 42.
    label_descriptions: Optional dict mapping concept names to descriptions.
    """

    def __init__(
        self,
        root: str = None,
        num_train_samples: int = 10000,
        num_test_samples: int = 2000,
        val_size: float = 0.1,
        img_size: int = 224,
        seed: int = 42,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.val_size = val_size
        self.label_descriptions = label_descriptions
        self.img_size = img_size
        self.seed = seed

        self.operators = ('+', '-', 'x', '/')

        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'mnist_arithmetic')
        self.root = root

        self.mnist_root = os.path.join(self.root, "mnist_data")

        filenames, concepts, annotations, graph = self.load()

        super().__init__(
            input_data=filenames,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name="MNISTArithmeticDataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        return []

    @property
    def processed_filenames(self) -> List[str]:
        return [
            f"filenames_Ntrain_{self.num_train_samples}_Ntest_{self.num_test_samples}_seed_{self.seed}.txt",
            f"concepts_Ntrain_{self.num_train_samples}_Ntest_{self.num_test_samples}_seed_{self.seed}.pt",
            "annotations.pt",
            "split_mapping.h5",
        ]

    def download(self):
        """Download MNIST dataset (handled by torchvision)."""
        datasets.MNIST(root=self.mnist_root, train=True, download=True)
        datasets.MNIST(root=self.mnist_root, train=False, download=True)

    def maybe_download(self):
        """Download and extract the dataset if needed."""
        super().maybe_download()

    def build(self):
        """Generate composite arithmetic images from both MNIST splits and save metadata."""

        self.maybe_download()

        logger.info(f"Generating MNIST arithmetic dataset "
                     f"(train={self.num_train_samples}, test={self.num_test_samples}, seed={self.seed})")

        img_dir = os.path.join(self.root_dir, "images")

        # Generate from MNIST train split
        train_filenames, train_concepts, train_tasks = _generate_arithmetic_data(
            img_dir=img_dir,
            mnist_root=self.mnist_root,
            train=True,
            num_samples=self.num_train_samples,
            img_size=self.img_size,
            seed=self.seed,
            filename_offset=0,
        )

        # Generate from MNIST test split (use different seed to avoid identical operator sequence)
        test_filenames, test_concepts, test_tasks = _generate_arithmetic_data(
            img_dir=img_dir,
            mnist_root=self.mnist_root,
            train=False,
            num_samples=self.num_test_samples,
            img_size=self.img_size,
            seed=self.seed + 1,
            filename_offset=self.num_train_samples,
        )

        # Combine all
        all_filenames = train_filenames + test_filenames
        all_concepts = train_concepts + test_concepts
        all_tasks = train_tasks + test_tasks

        cy = []
        for c, t in zip(all_concepts, all_tasks):
            cy.append(c + t)
        cy = torch.tensor(cy, dtype=torch.float32)

        cy_names = CONCEPT_NAMES + TASK_NAMES
        concept_metadata = {name: {'type': 'continuous'} for name in cy_names}
        cardinalities = tuple([1] * len(cy_names))

        annotations = Annotations({
            1: AxisAnnotation(
                labels=cy_names,
                cardinalities=cardinalities,
                metadata=concept_metadata,
            )
        })

        # Build split mapping: randomly split MNIST-train pool into train/val
        np.random.seed(self.seed)
        n_val = int(self.num_train_samples * self.val_size)
        perm = np.random.permutation(self.num_train_samples)
        val_indices = set(perm[:n_val].tolist())

        split_labels = []
        for i in range(self.num_train_samples):
            split_labels.append('val' if i in val_indices else 'train')
        for _ in range(self.num_test_samples):
            split_labels.append('test')

        # Save
        os.makedirs(self.root_dir, exist_ok=True)

        with open(self.processed_paths[0], 'w') as f:
            f.write('\n'.join(all_filenames))

        torch.save(cy, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        pd.Series(split_labels).to_hdf(self.processed_paths[3], key="split_mapping", mode="w")

        logger.info(f"MNIST arithmetic dataset saved to {self.root_dir} "
                     f"(train={self.num_train_samples - n_val}, val={n_val}, test={self.num_test_samples})")

    def load_raw(self):
        """Load raw processed files for the current split."""
        self.maybe_build()

        logger.info(f"Loading MNIST arithmetic dataset from {self.root_dir}")

        with open(self.processed_paths[0], 'r') as f:
            filenames = f.read().strip().split('\n')

        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = None

        return filenames, concepts, annotations, graph

    def load(self):
        return self.load_raw()

    def __getitem__(self, item):
        if self.embs_precomputed:
            x = self.input_data[item]
        else:
            filename = self.input_data[item]
            img_path = os.path.join(self.root_dir, "images", filename)
            img = Image.open(img_path).convert('RGB')
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        c = self.concepts[item]

        return {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

    @property
    def n_samples(self) -> int:
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)
