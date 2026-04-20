import os
import math
import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Optional
from PIL import Image
from tqdm import tqdm

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)

CONCEPT_NAMES = ['theta', 'phi']
TASK_NAMES = ['pendulum_x']


def _projection(phi, x_0, y_0, base=-0.5):
    """Calculate x intersection between line y - y_0 = tan(phi)(x - x_0) and y = base."""
    b = y_0 - x_0 * math.tan(phi)
    shade = (base - b) / math.tan(phi)
    return shade


def _generate_pendulum_data(root_dir, n_theta=100, n_phi=1000, seed=42):
    """Generate pendulum images and metadata.

    Args:
        root_dir: Directory to save images and metadata.
        n_theta: Number of theta angle steps.
        n_phi: Number of phi angle steps.
        seed: Random seed (used for train/val/test split cycling).

    Returns:
        Tuple of (all_filenames, all_concepts, all_tasks) as lists.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_dir = os.path.join(root_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    all_filenames = []
    all_concepts = []
    all_tasks = []

    count = 0
    for theta in tqdm(np.linspace(-200, 200, n_theta), desc="Generating pendulum images"):
        for phi in np.linspace(60, 140, n_phi):
            if phi == 100:
                continue

            theta_rad = theta * math.pi / 200.0
            phi_rad = phi * math.pi / 200.0

            # Pendulum ball coordinates
            x = 10 + 8 * math.sin(theta_rad)
            y = 10.5 - 8 * math.cos(theta_rad)

            # Draw pendulum scene
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            ball = plt.Circle((x, y), 1.5, color='firebrick')
            gun = plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3)

            light = _projection(phi_rad, 10, 10.5, 20.5)
            sun = plt.Circle((light, 20.5), 3, color='orange')

            ball_x = 10 + 9.5 * math.sin(theta_rad)
            ball_y = 10.5 - 9.5 * math.cos(theta_rad)

            mid = (_projection(phi_rad, 10.0, 10.5) + _projection(phi_rad, ball_x, ball_y)) / 2
            shade = max(3, abs(_projection(phi_rad, 10.0, 10.5) - _projection(phi_rad, ball_x, ball_y)))

            shadow = plt.Polygon(([mid - shade / 2.0, -0.5], [mid + shade / 2.0, -0.5]),
                                 color='black', linewidth=3)

            ax = plt.gca()
            ax.add_artist(gun)
            ax.add_artist(ball)
            ax.add_artist(sun)
            ax.add_artist(shadow)
            ax.set_xlim((0, 20))
            ax.set_ylim((-1, 21))
            plt.axis('off')

            fname = f"a_{round(float(theta), 4)}_{round(float(phi), 4)}.png"
            filepath = os.path.join(img_dir, fname)
            plt.savefig(filepath, dpi=96, transparent=False)
            plt.clf()

            all_filenames.append(fname)
            all_concepts.append([theta_rad, phi_rad])
            all_tasks.append([x])

            count += 1

    plt.close('all')
    return all_filenames, all_concepts, all_tasks


class PendulumDataset(ConceptDataset):
    """Procedurally generated pendulum scene dataset for regression.

    Each sample is a rendered image of a pendulum with a light source casting
    a shadow. The concepts are the pendulum angle (theta) and the light angle
    (phi), both continuous. The regression task is to predict the x-coordinate
    of the pendulum ball.

    Parameters
    ----------
    root : str, optional
        Root directory to store/load the dataset. If None, defaults to
        ``'./data/pendulum'``.
    n_theta : int, optional
        Number of theta angle steps for generation. Default: 100
    n_phi : int, optional
        Number of phi angle steps for generation. Default: 1000
    seed : int, optional
        Random seed for reproducibility. Default: 42
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None (all concepts).

    Attributes
    ----------
    input_data : list
        List of image filenames (images loaded on-the-fly).
    concepts : torch.Tensor
        Tensor of shape (n_samples, 3) containing [theta, phi, pendulum_x].

    Examples
    --------
    >>> from torch_concepts.data.datasets import PendulumDataset
    >>> dataset = PendulumDataset(root='./data/pendulum', n_theta=10, n_phi=10)
    >>> sample = dataset[0]
    >>> x = sample['inputs']['x']  # image tensor (C, H, W)
    >>> c = sample['concepts']['c']  # [theta, phi, pendulum_x]
    """

    def __init__(
        self,
        root: str = None,
        n_theta: int = 100,
        n_phi: int = 1000,
        seed: int = 42,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.seed = seed
        self.label_descriptions = label_descriptions

        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'pendulum')

        self.root = root

        filenames, concepts, annotations, graph = self.load()

        super().__init__(
            input_data=filenames,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name="PendulumDataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        return []

    @property
    def processed_filenames(self) -> List[str]:
        return [
            f"filenames_theta_{self.n_theta}_phi_{self.n_phi}.txt",
            f"concepts_theta_{self.n_theta}_phi_{self.n_phi}.pt",
            "annotations.pt",
        ]

    def download(self):
        """This dataset is procedurally generated."""
        pass

    def build(self):
        """Generate pendulum images and save metadata to disk."""
        logger.info(f"Generating pendulum dataset (n_theta={self.n_theta}, n_phi={self.n_phi})")

        filenames, concepts_list, tasks_list = _generate_pendulum_data(
            self.root_dir, self.n_theta, self.n_phi, self.seed
        )

        # Combine concepts and tasks: [theta, phi, pendulum_x]
        cy = []
        for c, t in zip(concepts_list, tasks_list):
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

        os.makedirs(self.root_dir, exist_ok=True)

        with open(self.processed_paths[0], 'w') as f:
            f.write('\n'.join(filenames))

        torch.save(cy, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])

        logger.info(f"Pendulum dataset saved to {self.root_dir}")

    def load_raw(self):
        self.maybe_build()
        logger.info(f"Loading pendulum dataset from {self.root_dir}")

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
