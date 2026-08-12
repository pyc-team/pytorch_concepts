from typing import Dict, List, Optional, Sequence, Union

import torch

from ..datasets.colormnist import ColorMNISTDataset

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..splitters import FixedIndicesSplitter
from ..utils import concat_datasets, resolve_size


class ColorMNISTDataModule(ConceptDataModule):
    """DataModule for the Color-MNIST dataset, with optional distribution shift.

    MNIST's train and test images are colourized separately and appended into
    one dataset, and the default split follows that boundary: validation is
    carved out of the train images, and the test split is exactly MNIST's own.
    ``test_size`` is therefore fixed by MNIST (10k images) unless a different
    ``splitter`` is passed.

    Because the two populations are coloured independently, ``coloring_test``
    is all it takes to study a **distribution shift**: the second colouring then
    appears in the test split and nowhere else. Left at None, both are coloured
    the same way and the dataset is simply all of MNIST.

    Parameters
    ----------
    root : str, optional
        Directory MNIST is downloaded to. Default: ``./data/mnist``.
    colors : sequence of str, optional
        Colour names to tint with. Default: ``('red', 'green')``
    coloring : str or dict, optional
        ``'random'`` or a ``{colour: digits}`` map, applied to the training
        population. Default: ``'random'``
    coloring_test : str or dict, optional
        Colouring of the test population. When given, the module holds two
        differently-coloured populations instead of one. Default: None
    seed : int, optional
        Random seed for the colour assignment and the split. Default: 42
    splitter : Splitter, optional
        Splitting strategy. Defaults to a :class:`FixedIndicesSplitter` on
        MNIST's own train/test boundary; pass ``RandomSplitter()`` to ignore it
        (which also dissolves any shift).
    val_size : int or float, optional
        Validation set size, taken from the train images. Default: 0.1
    test_size : int or float, optional
        Ignored by the default splitter, which uses MNIST's test split.
        Default: 0.2
    batch_size : int, optional
        Number of samples per batch. Default: 512
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None
    workers : int, optional
        Number of data loading workers. Default: 0
    **kwargs
        Forwarded to :class:`ConceptDataModule` (e.g. ``max_samples``,
        ``scalers``, ``pin_memory``). ``max_samples`` subsamples every split
        proportionally: the base class remaps the fixed indices onto the rows
        that survive.

    Examples
    --------
    >>> from torch_concepts.data import ColorMNISTDataModule
    >>>
    >>> dm = ColorMNISTDataModule(batch_size=64, max_samples=2000)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    A shift: low digits are red while training and green while testing, so a
    model reading the colour instead of the digit collapses on the test split.

    >>> dm = ColorMNISTDataModule(                       # doctest: +SKIP
    ...     coloring={'red': range(5), 'green': range(5, 10)},
    ...     coloring_test={'red': range(5, 10), 'green': range(5)},
    ... )

    See Also
    --------
    ColorMNISTDataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """

    def __init__(
        self,
        root: str = None,
        colors: Sequence[str] = ('red', 'green'),
        coloring: Union[str, Dict[str, Sequence[int]]] = 'random',
        coloring_test: Optional[Union[str, Dict[str, Sequence[int]]]] = None,
        seed: int = 42,
        splitter: Optional[Splitter] = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: Optional[List[str]] = None,
        workers: int = 0,
        **kwargs,
    ):
        # A dataset carries one colouring, so the two populations are built
        # separately and appended: MNIST's train images first, its test images
        # after. Without `coloring_test` they are coloured the same way and this
        # is simply all of MNIST; with it, the second colouring lives on its own
        # row range and the split below turns that into a distribution shift.
        train_population = ColorMNISTDataset(
            root=root,
            train=True,
            colors=colors,
            coloring=coloring,
            seed=seed,
            concept_subset=concept_subset,
        )
        test_population = ColorMNISTDataset(
            root=root,
            train=False,
            colors=colors,
            coloring=coloring if coloring_test is None else coloring_test,
            seed=seed + 1,
            concept_subset=concept_subset,
        )
        n_train = train_population.n_samples
        dataset = concat_datasets(train_population, test_population)

        if splitter is None:
            # Keep MNIST's own partition
            n_val = resolve_size(val_size, n_train)
            order = torch.randperm(
                n_train, generator=torch.Generator().manual_seed(seed)
            ).tolist()
            splitter = FixedIndicesSplitter(
                train_idxs=order[n_val:],
                val_idxs=order[:n_val],
                test_idxs=range(n_train, dataset.n_samples),
            )

        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            workers=workers,
            splitter=splitter,
            seed=seed,
            **kwargs,
        )
