from ..datasets.cebab import CEBaBDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters.native import NativeSplitter


class CEBaBDataModule(ConceptDataModule):
    """DataModule for CEBaB (Concept Bottleneck Model).

    Handles data loading, splitting, and batching for the CEBaB dataset
    with support for concept-based learning.  CEBaB provides official
    train / val / test splits,
    so :class:`~torch_concepts.data.splitters.NativeSplitter` is used by
    default.

    .. note::
        CEBaB must be downloaded before use (it is automatically done in :class:`~torch_concepts.data.datasets.CEBaBDataset`).
        See :class:`~torch_concepts.data.datasets.CEBaBDataset` for instructions.

    Parameters
    ----------
    root : str, optional
        Root directory containing the CEBaB files.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``NativeSplitter()`` (uses the official
        train / val / test splits).
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
    backbone : BackboneType, optional
        Backbone model for feature extraction (e.g. ``'bert-base-uncased'``).
        Default: ``bert-base-uncased``.
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings.  Default: ``True``.
    force_recompute : bool, optional
        Recompute embeddings even if a cache exists.  Default: ``False``.
    concepts_type : str, optional
        Whether to use 'discrete' or 'continuous' representations for concepts and target.  Default: 'discrete'.
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.

    Examples
    --------
    >>> from torch_concepts.data import CEBaBDataModule
    >>>
    >>> dm = CEBaBDataModule(
    ...     root="./data/CEBaB",
    ...     batch_size=64,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    CEBaBDataset : The underlying dataset class.
    ConceptDataModule : Parent class with common datamodule functionality.
    """

    def __init__(
        self,
        root: str = None,
        splitter: Splitter = NativeSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = 'bert-base-uncased',
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concepts_type: str = 'discrete',
        workers: int = 0,
        **kwargs,
    ):
        dataset = CEBaBDataset(
            root=root,
            concepts_type=concepts_type,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            backbone=backbone,
            backbone_input_type = 'text',
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter,
        )
