from ..datasets.cub import CUBDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters.native import NativeSplitter


class CUBDataModule(ConceptDataModule):
    """DataModule for CUB-200-2011 (Caltech-UCSD Birds).

    Handles data loading, splitting, and batching for the CUB-200-2011 dataset
    with support for concept-based learning.  CUB-200-2011 provides official
    train / val / test splits via the Koh et al. pre-processed pickle files,
    so :class:`~torch_concepts.data.splitters.NativeSplitter` is used by
    default.

    .. note::
        CUB-200-2011 must be **manually downloaded** before use.
        See :class:`~torch_concepts.data.datasets.CUBDataset` for instructions.

    Parameters
    ----------
    root : str, optional
        Root directory containing ``class_attr_data_10/`` and
        ``CUB_200_2011/``.  Default: ``None`` (auto-creates ``./data/CUB200``).
    image_size : int, optional
        Side length (px) to resize images to.  Default: 224.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``NativeSplitter()`` (uses the official
        train / val / test splits from the pickle files).
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
        Subset of concept names to retain.  Default: ``None`` (all 113).
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.

    Examples
    --------
    >>> from torch_concepts.data import CUBDataModule
    >>>
    >>> dm = CUBDataModule(
    ...     root="./data/CUB200",
    ...     backbone="resnet50",
    ...     precompute_embs=True,
    ...     batch_size=64,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    CUBDataset : The underlying dataset class.
    ConceptDataModule : Parent class with common datamodule functionality.
    """

    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        splitter: Splitter = NativeSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = CUBDataset(
            root=root,
            image_size=image_size,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter,
        )
