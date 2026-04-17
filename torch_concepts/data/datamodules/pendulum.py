from ..datasets import PendulumDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters import RandomSplitter


class PendulumDataModule(ConceptDataModule):
    """DataModule for Pendulum dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the procedurally generated
    pendulum scene dataset. Supports precomputing backbone embeddings and
    flexible train/val/test splitting strategies.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or will be generated.
        Default: None (auto-creates ``./data/pendulum``).
    n_theta : int, optional
        Number of theta angle steps for generation. Default: 100
    n_phi : int, optional
        Number of phi angle steps for generation. Default: 1000
    seed : int, optional
        Random seed for reproducibility. Default: 42
    splitter : Splitter, optional
        Splitting strategy for train/val/test partitioning.
        Default: RandomSplitter()
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
        Subset of concept names to use. Default: None
    workers : int, optional
        Number of data loading workers. Default: 0

    Examples
    --------
    >>> from torch_concepts.data.datamodules import PendulumDataModule
    >>>
    >>> dm = PendulumDataModule(
    ...     root='./data/pendulum',
    ...     n_theta=10, n_phi=10,
    ...     batch_size=32, seed=42,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    PendulumDataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """

    def __init__(
        self,
        root: str = None,
        n_theta: int = 100,
        n_phi: int = 1000,
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
        dataset = PendulumDataset(
            root=root,
            n_theta=n_theta,
            n_phi=n_phi,
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
