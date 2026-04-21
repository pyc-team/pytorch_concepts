from typing import Dict, List, Optional

from ..datasets.dsprites_regression import DSpritesRegressionDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters import RandomSplitter


class DSpritesRegressionDataModule(ConceptDataModule):
    """DataModule for DSprites regression dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the DSprites dataset with
    sympy formula-based regression targets. Supports precomputing backbone
    embeddings and flexible train/val/test splitting strategies.

    Parameters
    ----------
    root : str, optional
        Root directory for caching. Default: None (``./data/dsprites_regression``).
    formulas : dict, optional
        Mapping from shape name to sympy formula string. Default: None (uses
        built-in defaults).
    seed : int, optional
        Random seed. Default: 42
    splitter : Splitter, optional
        Splitting strategy. Default: RandomSplitter()
    val_size : int or float, optional
        Validation set size. Default: 0.1
    test_size : int or float, optional
        Test set size. Default: 0.2
    batch_size : int, optional
        Number of samples per batch. Default: 512
    backbone : BackboneType, optional
        Backbone model for feature extraction. Default: None
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings. Default: True
    force_recompute : bool, optional
        If True, recompute embeddings even if cached. Default: False
    concept_subset : list of str, optional
        Subset of concept names to retain after loading. Default: None
    label_descriptions : dict, optional
        Optional dict mapping concept names to descriptions.
    workers : int, optional
        Number of data loading workers. Default: 0
    """

    def __init__(
        self,
        root: str = None,
        formulas: Optional[Dict[str, str]] = None,
        seed: int = 42,
        splitter: Splitter = RandomSplitter(),
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = DSpritesRegressionDataset(
            root=root,
            formulas=formulas,
            seed=seed,
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
