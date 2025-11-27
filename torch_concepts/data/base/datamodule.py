"""Base LightningDataModule for concept-based datasets.

Provides data splitting, scaling, embedding precomputation, and DataLoader
configuration for concept-based learning tasks.
"""

import os
import logging
from typing import Literal, Mapping, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import ConceptDataset

logger = logging.getLogger(__name__)

from ..backbone import get_backbone_embs
from ..splitters.random import RandomSplitter
from ...typing import BackboneType

StageOptions = Literal['fit', 'validate', 'test', 'predict']


class ConceptDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for concept-based datasets.
    
    Handles the complete data pipeline:
    1. Data splitting (train/val/test)
    2. Optional backbone embedding precomputation and caching
    3. Data scaling/normalization
    4. DataLoader creation with appropriate configurations

    Args:
        dataset (ConceptDataset): Complete dataset to be split.
        val_size (float, optional): Validation set fraction. Defaults to 0.1.
        test_size (float, optional): Test set fraction. Defaults to 0.2.
        batch_size (int, optional): Mini-batch size. Defaults to 64.
        backbone (BackboneType, optional): Feature extraction model. If provided
            with precompute_embs=True, embeddings are computed and cached. Defaults to None.
        precompute_embs (bool, optional): Cache backbone embeddings to disk for
            faster retrieval. Defaults to False.
        force_recompute (bool, optional): Recompute embeddings even if cached. 
            Defaults to False.
        scalers (Mapping, optional): Dict of custom scalers for data normalization. 
            Keys must match the target keys in the batch (e.g., 'input', 'concepts'). 
            If None, no scaling is applied. Defaults to None.
        splitter (object, optional): Custom splitter for train/val/test splits.
            If None, uses RandomSplitter. Defaults to None.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        pin_memory (bool, optional): Enable pinned memory for GPU. Defaults to False.
        
    Example:
        >>> from torch_concepts.data.dataset import MNISTDataset
        >>> from torchvision.models import resnet18
        >>> 
        >>> dataset = MNISTDataset(...)
        >>> backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        >>> 
        >>> datamodule = ConceptDataModule(
        ...     dataset=dataset,
        ...     val_size=0.1,
        ...     test_size=0.2,
        ...     batch_size=64,
        ...     backbone=backbone,
        ...     precompute_embs=True,  # Cache embeddings for faster training
        ...     workers=4
        ... )
        >>> 
        >>> datamodule.setup('fit')
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        val_size: float = 0.1,
        test_size: float = 0.2,
        batch_size: int = 64,
        backbone: BackboneType = None,     # optional backbone
        precompute_embs: bool = False,
        force_recompute: bool = False, # whether to recompute embeddings even if cached
        scalers: Optional[Mapping] = None, # optional custom scalers
        splitter: Optional[object] = None, # optional custom splitter
        workers: int = 0,
        pin_memory: bool = False
    ):
        super(ConceptDataModule, self).__init__()
        self.dataset = dataset

        # backbone and embedding precomputation
        self.backbone = backbone
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
        return self.n_samples
    
    def __getattr__(self, item):
        ds = self.__dict__.get('dataset')
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        scalers_str = ', '.join(self.scalers.keys())
        return (f"{self.__class__.__name__}(train_len={self.train_len}, val_len={self.val_len}, "
                f"test_len={self.test_len}, scalers=[{scalers_str}], batch_size={self.batch_size}, "
                f"n_features={self.n_features}, n_concepts={self.n_concepts})")

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def testset(self):
        return self._testset
    
    @trainset.setter
    def trainset(self, value):
        self._add_set('train', value)

    @valset.setter
    def valset(self, value):
        self._add_set('val', value)

    @testset.setter
    def testset(self, value):
        self._add_set('test', value)

    @property
    def train_len(self):
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        return len(self.testset) if self.testset is not None else None

    @property
    def n_samples(self) -> int:
        """Number of samples (i.e., items) in the dataset."""
        return len(self.dataset)
    
    @property
    def bkb_embs_filename(self) -> str:
        """Filename for precomputed embeddings based on backbone name."""
        return f"bkb_embs_{self.backbone.__class__.__name__}.pt" if self.backbone is not None else None

    def _add_set(self, split_type, _set):
        """
        Add a dataset or a sequence of indices as a specific split.
        Args:
            split_type: One of 'train', 'val', 'test'. 
            _set: A Dataset or a sequence of indices.
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

    def maybe_use_backbone_embs(self, precompute_embs: bool = False, backbone_device: Optional[str] = None, verbose: bool = True):
        if verbose:
            logger.info(f"Input shape: {tuple(self.dataset.input_data.shape)}")
        if precompute_embs:
            if self.backbone is not None:
                # Precompute embeddings with automatic caching
                embs = get_backbone_embs(
                    path=os.path.join(self.dataset.root_dir, self.bkb_embs_filename) if self.bkb_embs_filename else None,
                    dataset=self.dataset,
                    backbone=self.backbone,
                    batch_size=self.batch_size,
                    force_recompute=self.force_recompute,  # whether to recompute embeddings even if cached
                    workers=self.workers,
                    device=backbone_device,
                    verbose=verbose,
                )
                self.dataset.input_data = embs
                self.embs_precomputed = True
                if verbose:
                    logger.info(f"✓ Using embeddings. New input shape: {tuple(self.dataset.input_data.shape)}")
            else:
                self.embs_precomputed = False
                if verbose:
                    logger.warning("Warning: precompute_embs=True but no backbone provided. Using raw input data.")
        else:
            # Use raw input data without preprocessing
            self.embs_precomputed = False
            if verbose:
                logger.info("Using raw input data without backbone preprocessing.")
                if self.backbone is not None:
                    logger.info("Note: Backbone provided but precompute_embs=False. Using raw input data.")

    def preprocess(self, precompute_embs: bool = False, backbone_device: Optional[str] = None, verbose: bool = True):
        """
        Preprocess the data. This method can be overridden by subclasses to
        implement custom preprocessing logic.
        
        Args:
            precompute_embs: Whether to precompute embeddings using backbone.
            verbose: Whether to print detailed logging information.
        """
        # ----------------------------------
        # Preprocess data with backbone if needed
        # ----------------------------------
        self.maybe_use_backbone_embs(precompute_embs, backbone_device=backbone_device, verbose=verbose)

    def setup(
            self, 
            stage: StageOptions = None, 
            backbone_device: Optional[str] = None,
            verbose: Optional[bool] = True):
        """
        Prepare the data. This method is called by Lightning with both
        'fit' and 'test' stages.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
                (default :obj:`None`, which means both 'fit' and 'test' stages)
            verbose: Print detailed logging information during setup and preprocessing.
                Defaults to True.
        
        Note:
            When precompute_embs=True:
                - If cached embeddings exist, they will be loaded automatically
                - If not, embeddings will be computed and saved to cache
                - Cache location: dataset.root_dir/embeddings_{backbone_name}.pt
            
            When precompute_embs=False:
                - Uses the original input_data without any backbone preprocessing
                - Backbone is ignored even if provided
        """

        # ----------------------------------
        # Preprocess data with backbone if needed
        # ----------------------------------
        self.preprocess(
            precompute_embs=self.precompute_embs, 
            backbone_device=backbone_device,
            verbose=verbose)

        # ----------------------------------
        # Splitting
        # ----------------------------------
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
        """
        Get the dataloader for a specific split.
        Args:
            split: One of 'train', 'val', 'test', or None. If None, returns
                a dataloader for the whole dataset.
                (default :obj:`None`, which means the whole dataset)
            shuffle: Whether to shuffle the data. Only used if `split` is
                'train'.
                (default :obj:`False`)
            batch_size: Size of the mini-batches. If :obj:`None`, uses
                :obj:`self.batch_size`.
                (default :obj:`None`)
        Returns:
            A DataLoader for the requested split, or :obj:`None` if the
            requested split is not available.
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
        return self.get_dataloader('train', shuffle, batch_size)

    def val_dataloader(self, shuffle: bool = False,
                      batch_size: Optional[int] = None) -> Optional[DataLoader]:
        return self.get_dataloader('val', shuffle, batch_size)

    def test_dataloader(self, shuffle: bool = False,
                       batch_size: Optional[int] = None) -> Optional[DataLoader]:
        return self.get_dataloader('test', shuffle, batch_size)
