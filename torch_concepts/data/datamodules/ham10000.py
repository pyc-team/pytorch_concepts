from ..datasets.ham10000 import HAM10000Dataset

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter


class HAM10000DataModule(ConceptDataModule):
    """DataModule for HAM10000 dermoscopic skin lesions.

    Handles data loading, splitting, and batching for the HAM10000 dataset.
    HAM10000 ships no official train / val / test split, so the default
    ``splitter=None`` falls back to a random split governed by ``val_size``
    and ``test_size``.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or will be downloaded.
        Default: ``None`` (auto-creates ``./data/ham10000``).
    image_size : int, optional
        Side length (px) to resize images to.  Default: 224.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``None`` (random split using
        ``val_size`` and ``test_size``).
    val_size : int or float, optional
        Validation set size. If float, interpreted as a fraction.
        Default: 0.1.
    test_size : int or float, optional
        Test set size. If float, interpreted as a fraction.  Default: 0.2.
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
    concept_subset : list of str, optional
        Subset of concept names to retain.  Default: ``None`` (all four:
        ``diagnosis``, ``age``, ``sex``, ``localization``).
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.

    Examples
    --------
    >>> from torch_concepts.data import HAM10000DataModule
    >>>
    >>> dm = HAM10000DataModule(root="./data/ham10000", batch_size=64)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    HAM10000Dataset : The underlying dataset class.
    ConceptDataModule : Parent class with common datamodule functionality.
    """

    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        splitter: Splitter = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: list | None = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = HAM10000Dataset(
            root=root,
            image_size=image_size,
            concept_subset=concept_subset,
        )

        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            workers=workers,
            splitter=splitter,
            **kwargs,
        )
