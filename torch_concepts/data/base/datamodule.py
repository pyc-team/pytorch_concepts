"""
Base LightningDataModule for concept-based datasets.

This module provides the :class:`ConceptDataModule` class, which handles
the complete data pipeline for concept-based learning tasks, including
data splitting, embedding precomputation, and DataLoader creation.

Example
-------
>>> from torch_concepts import ImageBackbone
>>> from torch_concepts.data import ConceptDataModule, CelebADataset
>>>
>>> dataset = CelebADataset(root='./data/celeba')
>>> dm = ConceptDataModule(
...     dataset=dataset,
...     val_size=0.1,
...     test_size=0.2,
...     batch_size=64
... )
>>> dm.setup('fit')
>>> train_loader = dm.train_dataloader()
"""

import logging
from typing import Literal, Mapping, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import ConceptDataset

logger = logging.getLogger(__name__)

from ..splitters import RandomSplitter, NativeSplitter

StageOptions = Literal['fit', 'validate', 'test', 'predict']


class ConceptDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for concept-based datasets.

    Handles the data pipeline for concept-based learning:

    1. **Data splitting**: Train/validation/test splits using configurable splitters
    2. **Data scaling**: Optional normalization through configurable scalers
    3. **DataLoader creation**: Efficient data loading with proper configurations

    Backbone embedding precomputation is a separate, explicit step: call
    :meth:`precompute_embeddings` with a :class:`~torch_concepts.Backbone` before
    ``setup()``. Embeddings are cached to disk and reloaded on subsequent runs.

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
    max_samples : int or None, optional
        If set, truncate the dataset to its first ``max_samples`` rows at
        construction — everything downstream (embedding precomputation,
        splitting, loaders) sees only the subset. Useful for quick runs and
        examples. Default is None (use all samples).
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
    seed : int or None, optional
        Seed controlling the train/val/test **split** only, passed to the
        splitter. If None, the split is non-deterministic. Default is None.

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
    scalers : dict
        Dictionary of scalers for data normalization.
    splitter : object
        The splitter used for data splitting.

    Examples
    --------
    Basic usage with random splitting:

    >>> from torch_concepts.data import ToyDataset
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

    Optional precomputation of backbone embeddings (before setup):

    >>> from torch_concepts.data import ToyDataset
    >>> dataset = ToyDataset(dataset='xor', n_gen=1000)
    >>> from torch_concepts import ImageBackbone
    >>> dm = ConceptDataModule(dataset=image_dataset, batch_size=64)
    >>> dm.precompute_embeddings(ImageBackbone('resnet50'))  # computes or loads cache
    >>> dm.setup('fit')  # splitting only

    See Also
    --------
    torch_concepts.ImageBackbone : Feature extraction wrapper class.
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
        max_samples: Optional[int] = None,
        scalers: Optional[Mapping] = None,
        splitter: Optional[object] = None,
        workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None
    ):
        super(ConceptDataModule, self).__init__()
        # Truncate the dataset to its first `max_samples` rows (all downstream
        # steps — embedding precompute, splitting, loaders — see the subset).
        if max_samples is not None:
            dataset.input_data = dataset.input_data[:max_samples]
            dataset.concepts = dataset.concepts[:max_samples]
            if isinstance(splitter, NativeSplitter):
                raise ValueError(
                    "'max_samples' is incompatible with NativeSplitter. Please pass "
                    "splitter=None (-> RandomSplitter) or a compatible splitter that "
                    "does not use explicit indices."
                )
        self.dataset = dataset

        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

        # init scalers
        if scalers is not None:
            self.scalers = scalers
        else:
            self.scalers = {}
            
        # split seed: controls the train/val/test partition
        self.seed = seed

        # set splitter
        self.trainset = self.valset = self.testset = None
        if splitter is None:
            self.splitter = RandomSplitter(
                val_size=val_size,
                test_size=test_size,
                seed=seed
            )
        else:
            # propagate the split seed to seed-aware splitters that weren't
            # given one explicitly (e.g. a default RandomSplitter() instance).
            if getattr(splitter, "seed", "__unset__") is None:
                splitter.seed = seed
            self.splitter = splitter

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
        return (f"{self.__class__.__name__}(n_samples={self.n_samples}, "
                f"train_len={self.train_len}, val_len={self.val_len}, "
                f"test_len={self.test_len}, scalers=[{scalers_str}], "
                f"n_features={self.n_features}, n_concepts={self.n_concepts}, "
                f"batch_size={self.batch_size})")

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

    def precompute_embeddings(self, backbone, cache: bool = True,
                              cache_dir: Optional[str] = None, force: bool = False) -> None:
        """Precompute backbone embeddings on the underlying dataset.

        Explicit preprocessing step — call it *before* :meth:`setup`. Delegates
        to :meth:`ConceptDataset.precompute_embeddings` with this datamodule's
        ``batch_size`` and ``workers``. With ``cache=True`` (default) the
        embeddings are persisted to ``{cache_dir or dataset.root_dir}/{backbone.filename}``
        and loaded from there on subsequent calls.

        Parameters
        ----------
        backbone : Backbone
            Feature extractor to run over the dataset.
        cache : bool, default True
            Persist the embeddings to disk and reuse them across calls.
        cache_dir : str, optional
            Directory for the cache file. Defaults to the dataset's ``root_dir``.
        force : bool, default False
            Recompute even if a cache file exists.
        """
        self.dataset.precompute_embeddings(
            backbone,
            batch_size=self.batch_size,
            workers=self.workers,
            cache=cache,
            cache_dir=cache_dir,
            force=force,
        )

    def setup(self, stage: StageOptions = None) -> None:
        """Prepare the data splits for training, validation, or testing.

        Called by PyTorch Lightning with 'fit', 'validate', 'test', or
        'predict' stages. Handles splitting (and, in the future,
        scaler fitting)

        Parameters
        ----------
        stage : {'fit', 'validate', 'test', 'predict'}, optional
            The stage for which data is being prepared. If None, prepares
            data for all stages. Default is None.
        """
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
        # The concept dataset owns batch collation so every batch's ground-truth
        # concepts arrive as an annotated tensor (see ``ConceptDataset.collate``).
        # Splits are ``Subset`` wrappers, so read the collate off the base dataset.
        collate_fn = getattr(self.dataset, 'collate', None)
        return DataLoader(dataset,
                          batch_size=batch_size or self.batch_size,
                          shuffle=shuffle,
                          drop_last=split == 'train',
                          num_workers=self.workers,
                          pin_memory=pin_memory,
                          collate_fn=collate_fn)

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
