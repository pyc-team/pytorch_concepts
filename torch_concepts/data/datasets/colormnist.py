from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import torch

from .mnist import default_root, load_mnist
from ..base.dataset import ConceptDataset
from ..utils import colorize
from ...annotations import Annotations

#: Colour name -> the RGB channel ``colorize`` moves the intensity into.
CHANNELS = {'red': 0, 'green': 1, 'blue': 2}


class ColorMNISTDataset(ConceptDataset):
    """MNIST digits tinted with one colour each.

    Concepts:
        ``digit``: 10-way categorical, the digit drawn in the image.
        ``parity``: binary, 1 when the digit is even.
        ``color``: categorical over ``colors``, the tint of the digit.

    ``parity`` is a deterministic function of ``digit``, which the concept graph
    records as the single edge ``digit -> parity``. Whether ``color`` depends on
    the digit is up to ``coloring`` and is deliberately *not* in the graph. This
    could be used to build a distribution shift between the train and test splits.

    Args:
        root: Directory MNIST is downloaded to. Defaults to ``./data/mnist``.
        train: Whether to use the MNIST train split. Default ``True``.
        colors: Colour names to tint with, in concept-value order. Each must be
            one of ``'red'``, ``'green'``, ``'blue'``. Default ``('red', 'green')``.
        coloring: How a colour is assigned to each image.

            * ``'random'`` (default): drawn uniformly, independent of the digit —
              the unbiased setting.
            * a dict ``{colour: [digits]}``: the colour is *determined* by the
              digit, e.g. ``{'red': [0, 1, 2, 3, 4], 'green': [5, 6, 7, 8, 9]}``.
              Every digit must appear exactly once. Build a distribution shift by
              instantiating the train and test splits with different maps.
        seed: Seed for the random colour assignment. Default ``42``.
        concept_subset: Optional subset of concept names.

    Example:
        >>> from torch_concepts.data import ColorMNISTDataset
        >>>
        >>> # Unbiased: colour says nothing about the digit.
        >>> dataset = ColorMNISTDataset()                          # doctest: +SKIP
        >>> dataset.n_features                                     # doctest: +SKIP
        (3, 28, 28)
        >>>
        >>> # Shortcut: low digits are red, high digits green in training...
        >>> train = ColorMNISTDataset(                             # doctest: +SKIP
        ...     train=True,
        ...     coloring={'red': range(5), 'green': range(5, 10)},
        ... )
        >>> # ...and the other way round at test time.
        >>> test = ColorMNISTDataset(                              # doctest: +SKIP
        ...     train=False,
        ...     coloring={'red': range(5, 10), 'green': range(5)},
        ... )
    """

    def __init__(
        self,
        root: str = None,
        train: bool = True,
        colors: Sequence[str] = ('red', 'green'),
        coloring: Union[str, Dict[str, Sequence[int]]] = 'random',
        seed: int = 42,
        concept_subset: Optional[List[str]] = None,
    ):
        unknown = [c for c in colors if c not in CHANNELS]
        if unknown:
            raise ValueError(
                f"ColorMNISTDataset: unknown colors {unknown}; "
                f"pick from {list(CHANNELS)}."
            )

        self.root = root or default_root('mnist')
        images, digits = load_mnist(self.root, train)
        color_ids = self._assign_colors(digits, colors, coloring, seed)

        labels = ['digit', 'parity', 'color']
        graph = pd.DataFrame(0, index=labels, columns=labels)
        graph.loc['digit', 'parity'] = 1

        # ``colorize`` needs the RGB channel, while the concept value is the
        # position in ``colors`` (so its cardinality follows the palette).
        channels = torch.tensor([CHANNELS[c] for c in colors])[color_ids]

        super().__init__(
            input_data=colorize(images, channels),                # (N, 3, 28, 28)
            concepts=torch.stack(
                [digits, (digits % 2 == 0).long(), color_ids], dim=1
            ),
            annotations=Annotations(
                labels=labels,
                cardinalities=[10, 1, len(colors)],
                types=['categorical', 'binary', 'categorical'],
            ),
            graph=graph,
            concept_names_subset=concept_subset,
            name="colormnist",
        )

    @staticmethod
    def _assign_colors(
        digits: torch.Tensor,
        colors: Sequence[str],
        coloring: Union[str, Dict[str, Sequence[int]]],
        seed: int,
    ) -> torch.Tensor:
        """One colour index per image, either drawn at random or read off the digit."""
        if coloring == 'random':
            return torch.randint(
                len(colors), (len(digits),),
                generator=torch.Generator().manual_seed(seed),
            )
        if not isinstance(coloring, dict):
            raise TypeError(
                "ColorMNISTDataset: `coloring` must be 'random' or a "
                f"{{colour: digits}} dict, got {type(coloring).__name__}."
            )

        color_ids = torch.full((len(digits),), -1, dtype=torch.long)
        for index, color in enumerate(colors):
            assigned = torch.tensor(list(coloring.get(color, [])), dtype=torch.long)
            color_ids[torch.isin(digits, assigned)] = index
        if (color_ids < 0).any():
            missing = sorted(digits[color_ids < 0].unique().tolist())
            raise ValueError(
                f"ColorMNISTDataset: `coloring` assigns no colour to digits {missing}. "
                "Every digit present in the split must appear under exactly one colour."
            )
        return color_ids
