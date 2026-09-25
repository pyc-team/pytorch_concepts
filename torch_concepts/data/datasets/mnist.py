"""Greyscale MNIST datasets with concept annotations.

Both datasets wrap ``torchvision.datasets.MNIST``, which already downloads and
caches the raw IDX files, so nothing extra is written to disk: the images are
assembled in memory and handed to :class:`ConceptDataset`.

See Also
--------
torch_concepts.data.ColorMNISTDataset : the colorized variant
"""
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch

from torchvision.datasets import MNIST

from ..base.dataset import ConceptDataset
from ...annotations import Annotations


def load_mnist(root: str, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load raw MNIST, downloading it on first use.

    Args:
        root: Directory torchvision downloads MNIST into.
        train: Whether to use the train split (else the test split).

    Returns:
        Tuple of ``(images, digits)``: images of shape ``(N, 28, 28)`` scaled to
        ``[0, 1]``, and their integer digit labels of shape ``(N,)``.
    """
    mnist = MNIST(root=root, train=train, download=True)
    return mnist.data.float() / 255.0, mnist.targets.long()


def default_root(name: str) -> str:
    """``./data/<name>`` under the current working directory."""
    return os.path.join(os.getcwd(), 'data', name)


class MNISTEvenOddDataset(ConceptDataset):
    """MNIST with the digit as a concept and its parity as the task.

    Concepts:
        ``digit``: 10-way categorical, the digit drawn in the image.
        ``parity``: binary, 1 when the digit is even.

    ``parity`` is a deterministic function of ``digit``, which the concept graph
    records as the single edge ``digit -> parity``.

    Args:
        root: Directory MNIST is downloaded to. Defaults to ``./data/mnist``.
        train: Whether to use the MNIST train split. Default ``True``.
        concept_subset: Optional subset of concept names.

    Example:
        >>> from torch_concepts.data import MNISTEvenOddDataset
        >>> dataset = MNISTEvenOddDataset()                   # doctest: +SKIP
        >>> dataset.n_features, dataset.concept_names         # doctest: +SKIP
        ((1, 28, 28), ['parity', 'digit'])
    """

    def __init__(
        self,
        root: str = None,
        train: bool = True,
        concept_subset: Optional[List[str]] = None,
    ):
        self.root = root or default_root('mnist')
        images, digits = load_mnist(self.root, train)

        labels = ['digit', 'parity']
        graph = pd.DataFrame(0, index=labels, columns=labels)
        graph.loc['digit', 'parity'] = 1

        super().__init__(
            input_data=images.unsqueeze(1),                       # (N, 1, 28, 28)
            concepts=torch.stack([digits, (digits % 2 == 0).long()], dim=1),
            annotations=Annotations(
                labels=labels,
                cardinalities=[10, 1],
                types=['categorical', 'binary'],
            ),
            graph=graph,
            concept_names_subset=concept_subset,
            name="MNISTEvenOddDataset",
        )


class MNISTAdditionDataset(ConceptDataset):
    """Pairs of MNIST digits side by side, with their sum as the task.

    Image ``i`` is paired with image ``-i`` (i.e. the ``i``-th from the end) and
    the two are concatenated horizontally into a ``(1, 28, 56)`` image.

    Concepts:
        ``first_digit``, ``second_digit``: 10-way categorical, the two digits.
        ``sum``: 19-way categorical, their sum (0 to 18).

    The concept graph records that both digits determine the sum. For the
    partially-annotated variant of this task, hide the second digit with
    ``concept_subset=['first_digit', 'sum']``.

    Args:
        root: Directory MNIST is downloaded to. Defaults to ``./data/mnist``.
        train: Whether to use the MNIST train split. Default ``True``.
        concept_subset: Optional subset of concept names.

    Example:
        >>> from torch_concepts.data import MNISTAdditionDataset
        >>> dataset = MNISTAdditionDataset()                  # doctest: +SKIP
        >>> dataset.n_features                                # doctest: +SKIP
        (1, 28, 56)
    """

    def __init__(
        self,
        root: str = None,
        train: bool = True,
        concept_subset: Optional[List[str]] = None,
    ):
        self.root = root or default_root('mnist')
        images, digits = load_mnist(self.root, train)
        # Pair each image with its mirror-index partner, so the two operands are
        # different images without drawing any randomness.
        first, second = images, images.flip(0)
        first_digit, second_digit = digits, digits.flip(0)

        labels = ['first_digit', 'second_digit', 'sum']
        graph = pd.DataFrame(0, index=labels, columns=labels)
        graph.loc[['first_digit', 'second_digit'], 'sum'] = 1

        super().__init__(
            input_data=torch.cat([first, second], dim=-1).unsqueeze(1),  # (N, 1, 28, 56)
            concepts=torch.stack(
                [first_digit, second_digit, first_digit + second_digit], dim=1
            ),
            annotations=Annotations(
                labels=labels,
                cardinalities=[10, 10, 19],
                types=['categorical', 'categorical', 'categorical'],
            ),
            graph=graph,
            concept_names_subset=concept_subset,
            name="MNISTAdditionDataset",
        )
