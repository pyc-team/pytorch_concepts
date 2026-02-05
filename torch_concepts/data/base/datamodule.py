"""
Base LightningDataModule for concept-based datasets.

This module provides the :class:`ConceptDataModule` class, which handles
the complete data pipeline for concept-based learning tasks, including
data splitting, embedding precomputation, and DataLoader creation.

The datamodule integrates with the :class:`Backbone` class for optional
feature extraction and embedding caching, enabling efficient training
workflows with pre-trained models.

Example
-------
>>> from torch_concepts.data import ConceptDataModule
>>> from torch_concepts.data.datasets import CelebADataset
>>> 
>>> dataset = CelebADataset(root='./data')
>>> dm = ConceptDataModule(
...     dataset=dataset,
...     backbone='resnet50',
...     precompute_embs=True,
...     batch_size=64
... )
>>> dm.setup('fit')
>>> train_loader = dm.train_dataloader()
"""

import os
import torch
import logging
from typing import Literal, Mapping, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .dataset import ConceptDataset

logger = logging.getLogger(__name__)

from ..backbone import Backbone
from ..splitters.random import RandomSplitter
from ...typing import BackboneType

StageOptions = Literal['fit', 'validate', 'test', 'predict']


class ConceptDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for concept-based datasets.

    Handles the complete data pipeline for concept-based learning:

    1. **Data splitting**: Train/validation/test splits using configurable splitters
    2. **Embedding precomputation**: Optional backbone feature extraction with caching
    3. **Data scaling**: Optional normalization through configurable scalers
    4. **DataLoader creation**: Efficient data loading with proper configurations

    The datamodule automatically caches computed embeddings to disk, allowing
    fast reloading on subsequent runs without recomputation.

    Parameters
    ----------
    dataset : ConceptDataset
        Complete dataset to be split and processed.
    val_size : float, optional
        Validation set fraction (0.0 to 1.0). Default is 0.1.
    test_size : float, optional
        Test set fraction (0.0 to 1.0). Default is 0.2.
    batch_size : int, optional
        Mini-batch size for DataLoaders. Default is 64.
    backbone : str or None, optional
        Feature extraction model name. Can be:

        - **HuggingFace model**: 'facebook/dinov2-base', 'google/vit-base-patch16-224'
        - **torchvision model**: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'

        If provided with ``precompute_embs=True``, embeddings are computed
        and cached to disk. Default is None.
    precompute_embs : bool, optional
        If True and backbone is provided, precompute and cache backbone
        embeddings before training. Embeddings are saved to
        ``{dataset.root_dir}/{backbone_filename}.pt``. Default is False.
    force_recompute : bool, optional
        If True, recompute embeddings even if cached file exists.
        Useful when the dataset or backbone changes. Default is False.
    scalers : Mapping or None, optional
        Dictionary of custom scalers for data normalization. Keys should
        match target keys in the batch (e.g., 'input', 'concepts').
        If None, no scaling is applied. Default is None.
    splitter : object or None, optional
        Custom splitter for train/val/test splits. Must implement a
        ``split(dataset)`` method that sets ``train_idxs``, ``val_idxs``,
        and ``test_idxs`` attributes. If None, uses RandomSplitter with
        the specified ``val_size`` and ``test_size``. Default is None.
    workers : int, optional
        Number of subprocesses for data loading. 0 means data will be
        loaded in the main process. Default is 0.
    pin_memory : bool, optional
        If True, the data loader will copy Tensors into pinned memory
        before returning them. Useful for GPU training. Default is False.

    Attributes
    ----------
    dataset : ConceptDataset
        The underlying concept dataset.
    trainset : Subset or None
        Training subset after setup().
    valset : Subset or None
        Validation subset after setup().
    testset : Subset or None
        Test subset after setup().
    backbone : Backbone or None
        The backbone wrapper for feature extraction.
    scalers : dict
        Dictionary of scalers for data normalization.
    splitter : object
        The splitter used for data splitting.

    Examples
    --------
    Basic usage with random splitting:

    >>> from torch_concepts.data.datasets import ToyDataset
    >>> dataset = ToyDataset(dataset='xor', n_gen=1000)
    >>> dm = ConceptDataModule(
    ...     dataset=dataset,
    ...     val_size=0.1,
    ...     test_size=0.2,
    ...     batch_size=32
    ... )
    >>> dm.setup('fit')
    >>> print(f"Train: {dm.train_len}, Val: {dm.val_len}, Test: {dm.test_len}")
    Train: 700, Val: 100, Test: 200

    Using backbone for embedding precomputation:

    >>> dm = ConceptDataModule(
    ...     dataset=image_dataset,
    ...     backbone='resnet50',
    ...     precompute_embs=True,
    ...     batch_size=64,
    ...     workers=4
    ... )
    >>> dm.setup('fit')  # Computes and caches embeddings
    >>> # On subsequent runs, embeddings are loaded from cache

    Using HuggingFace backbone:

    >>> dm = ConceptDataModule(
    ...     dataset=image_dataset,
    ...     backbone='facebook/dinov2-base',
    ...     precompute_embs=True
    ... )

    See Also
    --------
    Backbone : Feature extraction wrapper class.
    ConceptDataset : Base dataset class for concept data.
    RandomSplitter : Default splitter for train/val/test splits.
    NativeSplitter : Splitter using dataset's native splits.
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        val_size: float = 0.1,
        test_size: float = 0.2,
        batch_size: int = 64,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        scalers: Optional[Mapping] = None,
        splitter: Optional[object] = None,
        workers: int = 0,
        pin_memory: bool = False
    ):
        super(ConceptDataModule, self).__init__()
        self.dataset = dataset
        
        # Initialize dataset's embs_precomputed flag
        self.dataset.embs_precomputed = False

        # Wrap backbone in Backbone class if provided
        self._backbone = Backbone(backbone) if backbone is not None else None
        self.precompute_embs = precompute_embs
        self.force_recompute = force_recompute

        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

        # init scalers
        if scalers is not None:
            self.scalers = scalers
        else:
            self.scalers = {}
            
        # set splitter
        self.trainset = self.valset = self.testset = None
        if splitter is not None:
            self.splitter = splitter
        else:
            self.splitter = RandomSplitter(
                val_size=val_size,
                test_size=test_size
            )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the underlying dataset.
        """
        return self.n_samples
    
    def __getattr__(self, item):
        """Delegate attribute access to the underlying dataset.

        Parameters
        ----------
        item : str
            Attribute name to access.

        Returns
        -------
        object
            The attribute value from the dataset.

        Raises
        ------
        AttributeError
            If the attribute is not found in the datamodule or dataset.
        """
        ds = self.__dict__.get('dataset')
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        """Return string representation of the datamodule.

        Returns
        -------
        str
            Formatted string with split lengths, scalers, batch size, and dimensions.
        """
        scalers_str = ', '.join(self.scalers.keys())
        return (f"{self.__class__.__name__}(train_len={self.train_len}, val_len={self.val_len}, "
                f"test_len={self.test_len}, scalers=[{scalers_str}], batch_size={self.batch_size}, "
                f"n_features={self.n_features}, n_concepts={self.n_concepts})")

    @property
    def trainset(self):
        """The training subset.

        Returns
        -------
        Subset or None
            Training data subset, or None if not yet set up.
        """
        return self._trainset

    @property
    def valset(self):
        """The validation subset.

        Returns
        -------
        Subset or None
            Validation data subset, or None if not yet set up.
        """
        return self._valset

    @property
    def testset(self):
        """The test subset.

        Returns
        -------
        Subset or None
            Test data subset, or None if not yet set up.
        """
        return self._testset
    
    @trainset.setter
    def trainset(self, value):
        """Set the training subset."""
        self._add_set('train', value)

    @valset.setter
    def valset(self, value):
        """Set the validation subset."""
        self._add_set('val', value)

    @testset.setter
    def testset(self, value):
        """Set the test subset."""
        self._add_set('test', value)

    @property
    def train_len(self):
        """Number of samples in the training set.

        Returns
        -------
        int or None
            Training set length, or None if not set up.
        """
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        """Number of samples in the validation set.

        Returns
        -------
        int or None
            Validation set length, or None if not set up.
        """
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        """Number of samples in the test set.

        Returns
        -------
        int or None
            Test set length, or None if not set up.
        """
        return len(self.testset) if self.testset is not None else None

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.dataset)
    
    @property
    def backbone(self) -> Optional[Backbone]:
        """The backbone model wrapper for feature extraction.

        Returns
        -------
        Backbone or None
            The backbone wrapper, or None if not configured.
        """
        return self._backbone

    def _add_set(self, split_type, _set):
        """Add a dataset or indices as a specific split.

        Parameters
        ----------
        split_type : str
            One of 'train', 'val', 'test'.
        _set : Dataset, list, tuple, or None
            A Dataset instance, a sequence of indices, or None.

        Raises
        ------
        AssertionError
            If split_type is not 'train', 'val', or 'test'.
            If _set is not a valid type.
        """
        assert split_type in ['train', 'val', 'test']
        split_type = '_' + split_type
        name = split_type + 'set'
        
        # If _set is None or already a Dataset, set it directly
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        else:
            # Otherwise, treat it as a sequence of indices
            indices = _set
            assert isinstance(indices, (list, tuple)), \
                f"type {type(indices)} of `{name}` is not a valid type. " \
                "It must be a dataset or a sequence of indices."
            
            # Create a Subset only if there are indices
            if len(indices) > 0:
                _set = Subset(self.dataset, indices)
            else:
                _set = None  # Empty split
            setattr(self, name, _set)

    def _maybe_precompute_backbone_embeddings(
        self, 
        device: Optional[str] = None, 
        verbose: bool = True
    ) -> None:
        """Precompute and cache backbone embeddings if configured.

        Handles all preprocessing logic based on configuration:

        - If ``precompute_embs=False``: Sets ``embs_precomputed=False`` on dataset
        - If ``precompute_embs=True``: Loads from cache or computes embeddings

        Embeddings are cached to ``{dataset.root_dir}/{backbone.filename}``.

        Parameters
        ----------
        device : str, optional
            Device for backbone computation ('cpu', 'cuda', etc.).
            If None, uses backbone's auto-detected device.
        verbose : bool, optional
            If True, print detailed logging information. Default is True.

        Raises
        ------
        ValueError
            If ``precompute_embs=True`` but no backbone model is provided.

        Notes
        -----
        When cached embeddings are found and ``force_recompute=False``,
        embeddings are loaded directly without recomputation.
        """
        if verbose:
            logger.info(f"Input shape: {tuple(self.dataset[0]['inputs']['x'].shape)}")
        
        # If not precomputing, just mark dataset and return
        if not self.precompute_embs:
            self.dataset.embs_precomputed = False
            if verbose:
                logger.info("Using raw input data without backbone preprocessing.")
            return
        
        # Precompute embeddings
        if self.backbone is None:
            raise ValueError("precompute_embs=True but no backbone model provided.")
        
        # If device is explicitly provided, override backbone's device
        if device is not None:
            self._backbone = Backbone(self.backbone.name, device=device)
        
        cache_path = os.path.join(self.dataset.root_dir, self.backbone.filename)
        
        # Load from cache or compute
        if os.path.exists(cache_path) and not self.force_recompute:
            if verbose:
                logger.info(f"Loading precomputed embeddings from {cache_path}")
            embs = torch.load(cache_path)
        else:
            embs = self._compute_embeddings(verbose=verbose)
            if verbose:
                logger.info(f"Saving embeddings to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(embs, cache_path)
        
        # Update dataset with precomputed embeddings
        self.dataset.input_data = embs
        self.dataset.embs_precomputed = True
        
        if verbose:
            logger.info(f"Embeddings precomputed using {self.backbone}. "
                        f"New input shape: {tuple(embs.shape)}")

    def _compute_embeddings(self, verbose: bool = True) -> torch.Tensor:
        """Compute embeddings for the entire dataset using the backbone.

        Iterates through the dataset in batches and extracts embeddings
        using the configured backbone model.

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress information and show tqdm progress bar.
            Default is True.

        Returns
        -------
        torch.Tensor
            Stacked embeddings with shape (n_samples, embedding_dim),
            where embedding_dim depends on the backbone model.
        """
        if verbose:
            model_type = "HuggingFace" if self.backbone.is_huggingface else "torchvision"
            logger.info(f"Using {model_type} backbone: {self.backbone.name}")
            logger.info(f"Device: {self.backbone.device}")
        
        def collate_fn(batch):
            images = [sample['inputs']['x'] for sample in batch]
            if not self.backbone.is_huggingface and isinstance(images[0], torch.Tensor):
                return torch.stack(images)
            return images
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=collate_fn
        )
        
        embeddings_list = []
        if verbose:
            logger.info("Precomputing embeddings with backbone...")
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
            for batch_data in iterator:
                embedding = self.backbone(batch_data)
                embeddings_list.append(embedding.cpu())

        return torch.cat(embeddings_list, dim=0)

    def setup(
            self, 
            stage: StageOptions = None, 
            backbone_device: Optional[str] = None,
            verbose: Optional[bool] = True) -> None:
        """Prepare the data for training, validation, or testing.

        This method is called by PyTorch Lightning with 'fit', 'validate',
        'test', or 'predict' stages. It handles:

        1. Backbone embedding precomputation (if configured)
        2. Data splitting using the configured splitter

        Parameters
        ----------
        stage : {'fit', 'validate', 'test', 'predict'}, optional
            The stage for which data is being prepared. If None, prepares
            data for all stages. Default is None.
        backbone_device : str, optional
            Device for backbone computation ('cpu', 'cuda', etc.).
            If None, auto-detects available hardware. Default is None.
        verbose : bool, optional
            If True, print detailed logging information during setup.
            Default is True.

        Notes
        -----
        **Embedding Caching Behavior:**

        When ``precompute_embs=True``:

        - If cached embeddings exist at ``{dataset.root_dir}/{backbone.filename}``,
          they are loaded automatically
        - If not, embeddings are computed using the backbone and saved to cache
        - Set ``force_recompute=True`` to always recompute

        When ``precompute_embs=False``:

        - Uses original ``input_data`` without backbone preprocessing
        - Backbone is ignored even if provided

        Examples
        --------
        >>> dm = ConceptDataModule(dataset, backbone='resnet50', precompute_embs=True)
        >>> dm.setup('fit')  # Computes/loads embeddings and creates splits
        >>> dm.setup('test', backbone_device='cuda:1')  # Use specific GPU
        """
        # Preprocess data with backbone if needed
        self._maybe_precompute_backbone_embeddings(device=backbone_device, verbose=verbose)

        # Splitting
        if self.splitter is not None:
            self.splitter.split(self.dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs

        # ----------------------------------
        # Fit scalers on training data only
        # ----------------------------------
        # TODO: enable scalers and transforms
        
        # if stage in ['fit', None]:
        #     for key, scaler in self.scalers.items():
        #         if not hasattr(self.dataset, key):
        #             raise RuntimeError(f"setup(): Cannot find attribute '{key}' in dataset")
            
        #     train_data = getattr(self.dataset, key)
        #     if isinstance(self.trainset, Subset):
        #         train_data = train_data[self.trainset.indices]
            
        #     scaler.fit(train_data, dim=0)
        #     self.dataset.add_scaler(key, scaler)



    def get_dataloader(self, 
                       split: Literal['train', 'val', 'test'] = None,
                       shuffle: bool = False,
                       batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """Get the DataLoader for a specific split.

        Parameters
        ----------
        split : {'train', 'val', 'test'}, optional
            Which split to create a DataLoader for. If None, returns a
            DataLoader for the entire dataset. Default is None.
        shuffle : bool, optional
            Whether to shuffle the data. Typically True only for training.
            Default is False.
        batch_size : int, optional
            Mini-batch size. If None, uses ``self.batch_size``.
            Default is None.

        Returns
        -------
        DataLoader or None
            DataLoader for the requested split, or None if the split
            is not available (e.g., empty split).

        Raises
        ------
        ValueError
            If split is not one of 'train', 'val', 'test', or None.

        Notes
        -----
        For training DataLoaders, ``drop_last=True`` is set to ensure
        consistent batch sizes across iterations.
        """
        if split is None:
            dataset = self.dataset
        elif split in ['train', 'val', 'test']:
            dataset = getattr(self, f'{split}set')
        else:
            raise ValueError("Argument `split` must be one of "
                             "'train', 'val', 'test', or None.")
        if dataset is None: 
            return None
        pin_memory = self.pin_memory if split == 'train' else None
        return DataLoader(dataset,
                          batch_size=batch_size or self.batch_size,
                          shuffle=shuffle,
                          drop_last=split == 'train',
                          num_workers=self.workers,
                          pin_memory=pin_memory)

    def train_dataloader(self, shuffle: bool = True,
                        batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """Get the training DataLoader.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data. Default is True.
        batch_size : int, optional
            Mini-batch size. If None, uses ``self.batch_size``.

        Returns
        -------
        DataLoader or None
            Training DataLoader, or None if trainset is not available.
        """
        return self.get_dataloader('train', shuffle, batch_size)

    def val_dataloader(self, shuffle: bool = False,
                      batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """Get the validation DataLoader.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data. Default is False.
        batch_size : int, optional
            Mini-batch size. If None, uses ``self.batch_size``.

        Returns
        -------
        DataLoader or None
            Validation DataLoader, or None if valset is not available.
        """
        return self.get_dataloader('val', shuffle, batch_size)

    def test_dataloader(self, shuffle: bool = False,
                       batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """Get the test DataLoader.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data. Default is False.
        batch_size : int, optional
            Mini-batch size. If None, uses ``self.batch_size``.

        Returns
        -------
        DataLoader or None
            Test DataLoader, or None if testset is not available.
        """
        return self.get_dataloader('test', shuffle, batch_size)
