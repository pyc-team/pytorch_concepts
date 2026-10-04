import numpy as np

from ..datasets.ham10000 import HAM10000Dataset

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..splitters import FixedIndicesSplitter
from ..utils import resolve_size


def _lesion_grouped_split(lesion_ids, val_size, test_size, seed):
    """Assign whole lesions to splits, so no lesion spans two of them.

    Sizes are applied to lesions rather than images. Three quarters of
    HAM10000 lesions have a single image, so a fraction of lesions lands
    within ~0.2% of the same fraction of images.
    """
    lesions = np.unique(lesion_ids)
    np.random.default_rng(seed).shuffle(lesions)
    n_test = resolve_size(test_size, len(lesions))
    n_val = resolve_size(val_size, len(lesions))

    test = np.isin(lesion_ids, lesions[:n_test])
    val = np.isin(lesion_ids, lesions[n_test:n_test + n_val])
    return (
        np.flatnonzero(~test & ~val).tolist(),
        np.flatnonzero(val).tolist(),
        np.flatnonzero(test).tolist(),
    )


class HAM10000DataModule(ConceptDataModule):
    """DataModule for HAM10000 dermoscopic skin lesions.

    Handles data loading, splitting, and batching for the HAM10000 dataset.

    HAM10000 ships no official split, and 1,956 of its 7,470 lesions are
    photographed more than once -- 45% of all images. An image-level split
    therefore puts the same physical lesion in both train and test. The
    default splitter groups by ``lesion_id`` so a lesion never spans two
    splits. Pass an explicit splitter, such as
    :class:`~torch_concepts.data.splitters.RandomSplitter`, to opt out.

    .. note::
        The grouped split does not stratify by ``diagnosis``, which is highly
        imbalanced (67% ``nv``, 1% ``df``), so the rarest classes vary between
        seeds.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or will be downloaded.
        Default: ``None`` (auto-creates ``./data/ham10000``).
    image_size : int, optional
        Side length (px) to resize images to.  Default: 224.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``None``, which builds a
        :class:`~torch_concepts.data.splitters.FixedIndicesSplitter` holding a
        lesion-grouped split sized by ``val_size`` and ``test_size``.
    val_size : int or float, optional
        Validation set size. If float, a fraction; if int, a number of
        lesions.  Default: 0.1.
    test_size : int or float, optional
        Test set size. If float, a fraction; if int, a number of lesions.
        Default: 0.2.
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
    concept_subset : list of str, optional
        Subset of concept names to retain.  Default: ``None`` (all four:
        ``diagnosis``, ``age``, ``sex``, ``localization``).
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.
    seed : int, optional
        Seed for the lesion-grouped split and for ``max_samples``
        subsampling.  Default: ``None`` (non-deterministic).

    Examples
    --------
    >>> from torch_concepts.data import HAM10000DataModule
    >>>
    >>> dm = HAM10000DataModule(root="./data/ham10000", batch_size=64, seed=42)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    Opt out of lesion grouping:

    >>> from torch_concepts.data.splitters import RandomSplitter
    >>>
    >>> dm = HAM10000DataModule(splitter=RandomSplitter(test_size=0.2, seed=42))

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
        seed: int | None = None,
        **kwargs,
    ):
        dataset = HAM10000Dataset(
            root=root,
            image_size=image_size,
            concept_subset=concept_subset,
        )

        if splitter is None:
            splitter = FixedIndicesSplitter(
                *_lesion_grouped_split(
                    dataset.metadata["lesion_id"].to_numpy(),
                    val_size=val_size,
                    test_size=test_size,
                    seed=seed,
                )
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
