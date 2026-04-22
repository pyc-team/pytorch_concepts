from ..datasets.cebab import CEBaBDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters.native import NativeSplitter


class CEBaBDataModule(ConceptDataModule):
    """DataModule for CEBaB (Causal Estimation of Sentiment Polarity).

    Handles data loading, splitting, and batching for the CEBaB dataset with
    support for concept-based learning.  Uses the native train / val / test
    split provided by the dataset authors.

    Parameters
    ----------
    root : str, optional
        Root directory for caching processed artefacts.
        Default: ``None`` (auto-creates ``./data/cebab``).
    pre_trained_model_name : str, optional
        HuggingFace tokeniser / backbone model name.
        Default: ``'bert-base-uncased'``.
    max_length : int, optional
        Maximum token sequence length.  Default: 512.
    splitter : Splitter, optional
        Splitting strategy.  Default: ``NativeSplitter()`` (uses the
        author-provided train / val / test splits).
    batch_size : int, optional
        Number of samples per batch.  Default: 512.
    backbone : BackboneType, optional
        Backbone for precomputing text embeddings.  Default: ``None``.
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings.  Default: ``False``.
    force_recompute : bool, optional
        Recompute embeddings even if a cache exists.  Default: ``False``.
    concept_subset : list of str, optional
        Subset of concept names to retain.  Default: ``None`` (all 5).
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    workers : int, optional
        Number of data-loading worker processes.  Default: 0.

    Examples
    --------
    >>> from torch_concepts.data import CEBaBDataModule
    >>>
    >>> dm = CEBaBDataModule(
    ...     root="./data/cebab",
    ...     batch_size=128,
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
        pre_trained_model_name: str = "bert-base-uncased",
        max_length: int = 512,
        splitter: Splitter = NativeSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs,
    ):
        dataset = CEBaBDataset(
            root=root,
            pre_trained_model_name=pre_trained_model_name,
            max_length=max_length,
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
