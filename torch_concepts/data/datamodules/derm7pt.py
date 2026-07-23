from ..datasets.derm7pt import Derm7ptDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters import NativeSplitter


class Derm7ptDataModule(ConceptDataModule):
    """DataModule for Derm7pt dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the Derm7pt dataset. Supports precomputing backbone embeddings
    and flexible train/val/test splitting strategies.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility in data splitting and sampling.
    name : str
        Dataset identifier used for caching and logging purposes.
    root : str
        Root directory where the Derm7pt dataset is stored or will be downloaded.
    image_type : str, optional
        Type of images to use ('derm' for dermoscopic, 'clinic' for clinical images). Default: 'derm'
    group_infrequent_classes : bool, optional
        Whether to group infrequent classes into a single 'other' class. Default: True
    splitter : Splitter, optional
        Splitting strategy for train/val/test partitioning. Default: NativeSplitter()
        which uses Derm7pt's native split.
    val_size : int or float, optional
        Validation set size. If float, interpreted as fraction of training data.
        If int, interpreted as absolute number of samples. Default: 0.1
    test_size : int or float, optional
        Test set size. If float, interpreted as fraction of data.
        If int, interpreted as absolute number of samples. Default: 0.2
    batch_size : int, optional
        Number of samples per batch. Default: 512
    backbone : BackboneType, optional
        Backbone model for feature extraction (e.g., InceptionV3). If provided,
        can be used to precompute embeddings. Default: InceptionV3 pretrained on ImageNet
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings for faster training.
        Requires backbone to be specified. Default: True
    force_recompute : bool, optional
        If True, recompute embeddings even if cached version exists. Default: False
    concept_subset : list of str, optional
        Subset of concept/attribute names to use. If None, uses all concepts in Derm7pt dataset.
        Default: None
    label_descriptions : dict, optional
        Dictionary mapping attribute names to human-readable descriptions.
        Default: None
    workers : int, optional
        Number of worker processes for data loading. Default: 0 (main process only)
    **kwargs
        Additional arguments passed to parent ConceptDataModule.

    Attributes
    ----------
    dataset : Derm7ptDataset
        The underlying Derm7pt dataset instance.
    train_dataset : Dataset
        Training split of the dataset.
    val_dataset : Dataset
        Validation split of the dataset.
    test_dataset : Dataset
        Test split of the dataset.

    Examples
    --------
    Basic usage with default settings:

    >>> from torch_concepts.data import Derm7ptDataModule
    >>> 
    >>> dm = Derm7ptDataModule(
    ...     root='./data/derm7pt',
    ...     batch_size=64
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    With backbone for precomputed embeddings:

    >>> from torchvision.models import inception_v3
    >>> 
    >>> backbone = inception_v3(pretrained=True)
    >>> dm = Derm7ptDataModule(
    ...     root='./data/derm7pt',
    ...     backbone=backbone,
    ...     precompute_embs=True,
    ...     concept_subset=['pigmentation', 'pigment_network', 'vascular_structures'], # use a subset of concepts
    ... )

    See Also
    --------
    Derm7ptDataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """
    
    def __init__(
        self,
        root: str = None, # root directory for dataset
        image_type: str = "derm", # type of images to use ('derm' or 'clinic')
        group_infrequent_classes: bool = True, # whether to group infrequent classes
        splitter: Splitter = NativeSplitter(),
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = "InceptionV3",
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = Derm7ptDataset(
            root=root,
            image_type=image_type,
            group_infrequent_classes=group_infrequent_classes,
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
            splitter=splitter,
            **kwargs
        )
