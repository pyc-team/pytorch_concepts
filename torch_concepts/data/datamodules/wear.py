from typing import List, Optional

import torch

from ..datasets.wear import WearDataset

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..splitters import FixedIndicesSplitter
from ..utils import resolve_size


class WearDataModule(ConceptDataModule):
    """DataModule for the Wear dataset, split by machine.

    The windows of one machine overlap, so a random split of the windows
    would put the same steps of a machine in more than one split. The default
    split draws the machines instead: ``val_size`` and ``test_size`` count
    machines, and every window of a machine goes to the split of its machine.

    Parameters
    ----------
    root : str, optional
        Directory to store and load the generated data. Default: None
        (``./data/toy_datasets/wear``).
    n_gen : int, optional
        Number of machines. Default: 1500
    n_steps : int, optional
        Number of steps of each machine. Default: 20
    n_sensors : int, optional
        Number of sensor readings per step, besides the two conditions. Default: 14
    window : int, optional
        Steps per sample. Default: None (the whole series).
    stride : int, optional
        Steps from the start of one window to the start of the next.
        Default: None (``window``).
    seed : int, optional
        Random seed for the split. Default: 42
    generation_seed : int, optional
        Random seed for data generation. Default: 42
    splitter : Splitter, optional
        Splitting strategy. Default: None (by machine, as above).
    val_size : int or float, optional
        Validation set size, in machines. Default: 0.1
    test_size : int or float, optional
        Test set size, in machines. Default: 0.2
    batch_size : int, optional
        Number of samples per batch. Default: 512
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None
    workers : int, optional
        Number of data loading workers. Default: 0
    **kwargs
        Forwarded to :class:`ConceptDataModule` (e.g. ``max_samples``,
        ``scalers``, ``pin_memory``).

    Examples
    --------
    >>> from torch_concepts.data import WearDataModule
    >>>
    >>> dm = WearDataModule(n_gen=100, n_steps=12, window=4, stride=2, batch_size=32)
    >>> dm.setup()
    >>> batch = next(iter(dm.train_dataloader()))
    >>> batch["inputs"]["x"].shape
    torch.Size([32, 4, 16])

    See Also
    --------
    WearDataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """

    def __init__(
        self,
        root: str = None,
        n_gen: int = 1500,
        n_steps: int = 20,
        n_sensors: int = 14,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        seed: int = 42,
        generation_seed: int = 42,
        splitter: Optional[Splitter] = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: Optional[List[str]] = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = WearDataset(
            root=root,
            seed=generation_seed,
            n_gen=n_gen,
            n_steps=n_steps,
            n_sensors=n_sensors,
            window=window,
            stride=stride,
            concept_subset=concept_subset,
        )

        if splitter is None:
            # Draw the test and validation machines, then give each split all the
            # windows of its machines (`dataset.series` is the machine of each window).
            n_test, n_val = resolve_size(test_size, n_gen), resolve_size(val_size, n_gen)
            order = torch.randperm(n_gen, generator=torch.Generator().manual_seed(seed))
            test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]

            def windows(machines):
                return torch.where(torch.isin(dataset.series, machines))[0].tolist()

            splitter = FixedIndicesSplitter(
                train_idxs=windows(train), val_idxs=windows(val), test_idxs=windows(test)
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
