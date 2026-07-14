from ..datasets.awa2 import AWA2Dataset

from ..base.datamodule import ConceptDataModule
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
    image_size : int, optional
        Side length (px) to resize images to.  Default: 224.
    val_size : float, optional
        Fraction of samples for validation.  Default: 0.1.
    test_size : float, optional
        Fraction of samples for test.  Default: 0.2.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``RandomSplitter()`` (no official split
        exists for AwA2, so the datamodule owns the split).
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
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
    >>> dm = AWA2DataModule(root="./data/AWA2", batch_size=64)
    >>> dm.precompute_embeddings(Backbone("resnet50"))  # optional
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
        image_size: int = 224,
        val_size: float = 0.1,
        test_size: float = 0.2,
        splitter: Splitter = RandomSplitter(),
        batch_size: int = 512,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = AWA2Dataset(
            root=root,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            image_size=image_size,
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