from ..datasets.awa2 import AWA2Dataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters.random import RandomSplitter


class AWA2DataModule(ConceptDataModule):
    """DataModule for Animals with Attributes 2 (AwA2).

    Handles data loading, splitting, and batching for the AwA2 dataset with
    support for concept-based learning.  Since AwA2 has no official
    train/val/test split, splitting is performed by the datamodule using
    ``RandomSplitter`` by default.

    Parameters
    ----------
    root : str, optional
        Root directory where the AwA2 data is stored.
        Default: ``None`` (auto-creates ``./data/AWA2``).
    seed : int, optional
        Random seed for train / val / test split.  Default: 42.
    val_size : float, optional
        Fraction of samples for validation.  Default: 0.1.
    test_size : float, optional
        Fraction of samples for test.  Default: 0.2.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``RandomSplitter()`` (no official split
        exists for AwA2, so the datamodule owns the split).
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
    backbone : BackboneType, optional
        Backbone model for feature extraction (e.g. ``'resnet50'``).
        Default: ``None``.
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings.  Default: ``True``.
    force_recompute : bool, optional
        Recompute embeddings even if a cache exists.  Default: ``False``.
    concept_subset : list of str, optional
        Subset of concept names to retain.  Default: ``None`` (all 86).
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.

    Examples
    --------
    >>> from torch_concepts.data import AWA2DataModule
    >>>
    >>> dm = AWA2DataModule(
    ...     root="./data/AWA2",
    ...     backbone="resnet50",
    ...     precompute_embs=True,
    ...     batch_size=64,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    AWA2Dataset : The underlying dataset class.
    ConceptDataModule : Parent class with common datamodule functionality.
    """

    def __init__(
        self,
        root: str = None,
        seed: int = 42,
        val_size: float = 0.1,
        test_size: float = 0.2,
        splitter: Splitter = RandomSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = AWA2Dataset(
            root=root,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
        )

        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter,
        )
