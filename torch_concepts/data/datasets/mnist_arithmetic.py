import os
import random
import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)

CONCEPT_NAMES = ['first_digit', 'second_digit']
TASK_NAMES = ['result']


def _import_torchvision():
    """Lazily import torchvision, raising a clear error if it is not installed."""
    try:
        import torchvision as tv
        return tv
    except ImportError as exc:
        raise ImportError(
            "MNISTArithmeticDataset requires `torchvision`. "
            "Install it with: pip install torchvision"
        ) from exc


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
    # Fix the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    tv = _import_torchvision()
    mnist = tv.datasets.MNIST(root=mnist_root, train=train, download=False, transform=None)

    # Note: MNIST digits are 28x28.
    # The composite canvas is (84x28) before resizing to (img_size, img_size).
    resize_transform = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.Grayscale(num_output_channels=3),
    ])

    # REDUCED FONT SIZE: 20-24 is ideal for a 28x28 pixel block.
    try:
        font = ImageFont.truetype("arial.ttf", 22) 
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

        # Arithmetic Logic
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

        # --- COMPOSITE IMAGE GENERATION ---
        
        # 1. Create the main black canvas
        canvas = Image.new("L", (84, 28), color=0)
        canvas.paste(img1, (0, 0))

        # 2. Create the operator block
        op_canvas = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(op_canvas)

        if op == '-':
            # Draw a thick rectangle: [left, top, right, bottom]
            # This makes a nice, bold minus sign that won't look like a dot
            draw.rectangle([7, 13, 21, 14], fill=255)
        else:
            # Draw +, x, or / using the font
            # anchor="mm" ensures it's perfectly centered in the 28x28 block
            draw.text((14, 14), op, fill=255, font=font, anchor="mm")

        # 3. Assemble the equation
        canvas.paste(op_canvas, (28, 0))
        canvas.paste(img2, (56, 0))

        # Apply final resize and grayscale transforms
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
        return [
            "MNIST/raw/t10k-images-idx3-ubyte", # MNIST test images
            "MNIST/raw/t10k-labels-idx1-ubyte", # MNIST test labels
            "MNIST/raw/train-images-idx3-ubyte", # MNIST train images
            "MNIST/raw/train-labels-idx1-ubyte"  # MNIST train labels
        ]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            f"filenames_Ntrain_{self.num_train_samples}_Ntest_{self.num_test_samples}_seed_{self.seed}.txt",
            f"concepts_Ntrain_{self.num_train_samples}_Ntest_{self.num_test_samples}_seed_{self.seed}.pt",
            "annotations.pt",
            "split_mapping.h5",
        ]

    def download(self):
        """setup MNIST root and trigger MNIST download."""
        tv = _import_torchvision()
        tv.datasets.MNIST(root=self.root, train=True, download=True)
        tv.datasets.MNIST(root=self.root, train=False, download=True)

        # remove zipped raw files to save space
        for fname in self.raw_filenames:
            path = os.path.join(self.root, fname + ".gz")
            if os.path.exists(path):
                os.remove(path)

        raw_mnist_path = os.path.join(self.root, "MNIST/raw")
        logger.info(f"MNIST files downloaded to {raw_mnist_path}.")

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
            mnist_root=self.root,
            train=True,
            num_samples=self.num_train_samples,
            img_size=self.img_size,
            seed=self.seed,
            filename_offset=0,
        )

        # Generate from MNIST test split (use different seed to avoid identical operator sequence)
        test_filenames, test_concepts, test_tasks = _generate_arithmetic_data(
            img_dir=img_dir,
            mnist_root=self.root,
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
