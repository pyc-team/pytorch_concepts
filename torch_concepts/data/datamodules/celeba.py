from ..datasets import CelebADataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters import NativeSplitter


class CelebADataModule(ConceptDataModule):
    """DataModule for CelebA dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the CelebA (Large-scale 
    CelebFaces Attributes) dataset. Supports precomputing backbone embeddings
    and flexible train/val/test splitting strategies.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility in data splitting and sampling.
    name : str
        Dataset identifier used for caching and logging purposes.
    root : str
        Root directory where the CelebA dataset is stored or will be downloaded.
    splitter : Splitter, optional
        Splitting strategy for train/val/test partitioning. Default: NativeSplitter()
        which uses CelebA's native split.
    val_size : int or float, optional
        Validation set size. If float, interpreted as fraction of training data.
        If int, interpreted as absolute number of samples. Default: 0.1
    test_size : int or float, optional
        Test set size. If float, interpreted as fraction of data.
        If int, interpreted as absolute number of samples. Default: 0.2
    batch_size : int, optional
        Number of samples per batch. Default: 512
    backbone : BackboneType, optional
        Backbone model for feature extraction (e.g., ResNet, ViT). If provided,
        can be used to precompute embeddings. Default: None
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings for faster training.
        Requires backbone to be specified. Default: True
    force_recompute : bool, optional
        If True, recompute embeddings even if cached version exists. Default: False
    concept_subset : list of str, optional
        Subset of concept/attribute names to use. If None, uses all 40 CelebA
        attributes. Default: None
    label_descriptions : dict, optional
        Dictionary mapping attribute names to human-readable descriptions.
        Default: None
    workers : int, optional
        Number of worker processes for data loading. Default: 0 (main process only)
    **kwargs
        Additional arguments passed to parent ConceptDataModule.

    Attributes
    ----------
    dataset : CelebADataset
        The underlying CelebA dataset instance.
    train_dataset : Dataset
        Training split of the dataset.
    val_dataset : Dataset
        Validation split of the dataset.
    test_dataset : Dataset
        Test split of the dataset.

    Examples
    --------
    Basic usage with default settings:

    >>> from torch_concepts.data.datamodules import CelebADataModule
    >>> 
    >>> dm = CelebADataModule(
    ...     seed=42,
    ...     root='./data/celeba',
    ...     batch_size=64
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    With backbone for precomputed embeddings:

    >>> from torchvision.models import resnet18
    >>> 
    >>> backbone = resnet18(pretrained=True)
    >>> dm = CelebADataModule(
    ...     seed=42,
    ...     root='./data/celeba',
    ...     backbone=backbone,
    ...     precompute_embs=True,
    ...     concept_subset=['Smiling', 'Male', 'Young']
    ... )

    See Also
    --------
    CelebADataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """
    
    def __init__(
        self,
        root: str = None, # root directory for dataset
        splitter: Splitter = NativeSplitter(),
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
        dataset = CelebADataset(
            root=root,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions
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
            splitter=splitter
        )
