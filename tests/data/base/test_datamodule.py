"""Tests for torch_concepts.data.base.datamodule module."""

import pytest
import torch
import torch.nn as nn
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.datasets.toy import ToyDataset
from torch_concepts.annotations import Annotations
import tempfile
import os


@pytest.fixture
def toy_dataset():
    """Create a simple toy dataset for testing."""
    return ToyDataset(
        dataset='xor',
        n_gen=100,
        seed=42
    )


@pytest.fixture
def simple_backbone():
    """Create a simple backbone network."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 16)
    )


class TestConceptDataModuleInit:
    """Test ConceptDataModule initialization."""

    def test_basic_init(self, toy_dataset):
        """Test basic initialization."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2,
            batch_size=32
        )

        assert dm.dataset == toy_dataset
        assert dm.batch_size == 32
        assert dm.precompute_embs is False
        assert dm.backbone is None

    def test_with_backbone(self, toy_dataset, simple_backbone):
        """Test initialization with backbone."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone=simple_backbone,
            batch_size=16
        )

        assert dm.backbone is not None
        assert dm.batch_size == 16

    def test_with_scalers(self, toy_dataset):
        """Test initialization with custom scalers."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scalers = {
            'input': StandardScaler(),
            'concepts': StandardScaler()
        }

        dm = ConceptDataModule(
            dataset=toy_dataset,
            scalers=scalers
        )

        assert 'input' in dm.scalers
        assert 'concepts' in dm.scalers

    def test_custom_workers(self, toy_dataset):
        """Test initialization with custom worker count."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            workers=4,
            pin_memory=True
        )

        assert dm.workers == 4
        assert dm.pin_memory is True


class TestConceptDataModuleProperties:
    """Test ConceptDataModule properties."""

    def test_n_samples(self, toy_dataset):
        """Test n_samples property."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.n_samples == 100

    def test_len(self, toy_dataset):
        """Test __len__ method."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert len(dm) == 100

    def test_getattr_delegation(self, toy_dataset):
        """Test attribute delegation to dataset."""
        dm = ConceptDataModule(dataset=toy_dataset)

        # These should be delegated to the dataset
        assert hasattr(dm, 'n_features')
        assert hasattr(dm, 'n_concepts')
        assert dm.n_features == toy_dataset.n_features
        assert dm.n_concepts == toy_dataset.n_concepts

    def test_getattr_missing(self, toy_dataset):
        """Test that missing attributes raise AttributeError."""
        dm = ConceptDataModule(dataset=toy_dataset)

        with pytest.raises(AttributeError):
            _ = dm.nonexistent_attribute

    def test_bkb_embs_filename(self, toy_dataset, simple_backbone):
        """Test backbone embeddings filename generation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone=simple_backbone
        )

        assert dm.bkb_embs_filename is not None
        assert 'Sequential' in dm.bkb_embs_filename

    def test_bkb_embs_filename_no_backbone(self, toy_dataset):
        """Test backbone embeddings filename when no backbone."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.bkb_embs_filename is None


class TestConceptDataModuleSetup:
    """Test ConceptDataModule setup method."""

    def test_setup_fit(self, toy_dataset):
        """Test setup with fit stage."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup('fit')

        assert dm.trainset is not None
        assert dm.valset is not None
        assert dm.testset is not None

        # Check sizes
        assert dm.train_len > 0
        assert dm.val_len > 0
        assert dm.test_len > 0

        # Total should equal original dataset
        assert dm.train_len + dm.val_len + dm.test_len == 100

    def test_setup_test(self, toy_dataset):
        """Test setup with test stage."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            test_size=0.2
        )

        dm.setup('test')

        assert dm.testset is not None
        assert dm.test_len > 0

    def test_split_sizes(self, toy_dataset):
        """Test that split sizes are correct."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup('fit')

        # With 100 samples, 0.2 test should give ~20, 0.1 val should give ~10
        assert dm.test_len == pytest.approx(20, abs=2)
        assert dm.val_len == pytest.approx(10, abs=2)
        assert dm.train_len == pytest.approx(70, abs=2)


class TestConceptDataModuleDataLoaders:
    """Test ConceptDataModule dataloader methods."""

    def test_train_dataloader(self, toy_dataset):
        """Test train dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit')

        loader = dm.train_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_val_dataloader(self, toy_dataset):
        """Test validation dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit')

        loader = dm.val_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_test_dataloader(self, toy_dataset):
        """Test test dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('test')

        loader = dm.test_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_dataloader_iteration(self, toy_dataset):
        """Test that dataloaders can be iterated."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit')

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert 'inputs' in batch
        assert 'concepts' in batch
        assert 'x' in batch['inputs']
        assert 'c' in batch['concepts']

        # Check batch sizes
        assert batch['inputs']['x'].shape[0] <= 16
        assert batch['concepts']['c'].shape[0] <= 16


class TestConceptDataModuleRepr:
    """Test ConceptDataModule __repr__ method."""

    def test_repr_before_setup(self, toy_dataset):
        """Test repr before setup."""
        dm = ConceptDataModule(dataset=toy_dataset)
        repr_str = repr(dm)

        assert 'ConceptDataModule' in repr_str
        assert 'train_len=None' in repr_str
        assert 'val_len=None' in repr_str
        assert 'test_len=None' in repr_str

    def test_repr_after_setup(self, toy_dataset):
        """Test repr after setup."""
        dm = ConceptDataModule(dataset=toy_dataset)
        dm.setup('fit')
        repr_str = repr(dm)

        assert 'ConceptDataModule' in repr_str
        assert 'train_len=' in repr_str
        assert 'val_len=' in repr_str
        assert 'test_len=' in repr_str
        assert 'train_len=None' not in repr_str


class TestConceptDataModuleScalers:
    """Test ConceptDataModule with scalers."""

    def test_scaler_initialization(self, toy_dataset):
        """Test that scalers are properly initialized in the datamodule."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        dm = ConceptDataModule(
            dataset=toy_dataset,
            scalers={'input': scaler}
        )

        # Check that scalers are stored correctly
        assert 'input' in dm.scalers
        assert isinstance(dm.scalers['input'], StandardScaler)


class TestConceptDataModuleEdgeCases:
    """Test edge cases for ConceptDataModule."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        small_dataset = ToyDataset(dataset='xor', n_gen=10, seed=42)

        dm = ConceptDataModule(
            dataset=small_dataset,
            val_size=0.2,
            test_size=0.2,
            batch_size=2
        )

        dm.setup('fit')

        assert dm.train_len + dm.val_len + dm.test_len == 10

    def test_zero_val_size(self):
        """Test with zero validation size."""
        dataset = ToyDataset(dataset='xor', n_gen=50, seed=42)

        dm = ConceptDataModule(
            dataset=dataset,
            val_size=0.0,
            test_size=0.2,
            batch_size=8
        )

        dm.setup('fit')

        assert dm.val_len == 0 or dm.val_len is None or dm.valset is None

    def test_large_batch_size(self, toy_dataset):
        """Test with batch size close to dataset size."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=50,  # Half of dataset size
            val_size=0.1,
            test_size=0.1
        )

        dm.setup('fit')
        loader = dm.train_dataloader()

        # Should still work - with 80 samples and batch size 50, we get 1 batch
        # (Note: drop_last=True, so the last partial batch is dropped)
        batches = list(loader)
        # With ~80 training samples and batch_size=50, we should get 1 full batch
        assert len(batches) >= 1
        if len(batches) > 0:
            assert batches[0]['inputs']['x'].shape[0] == 50


class TestConceptDataModuleBackbone:
    """Test ConceptDataModule with backbone embeddings."""

    def test_precompute_embs_flag(self, toy_dataset, simple_backbone):
        """Test precompute_embs flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Modify dataset to use temp directory
            toy_dataset.root = tmpdir

            dm = ConceptDataModule(
                dataset=toy_dataset,
                backbone=simple_backbone,
                precompute_embs=True,
                batch_size=16
            )

            assert dm.precompute_embs is True
            assert dm.backbone is not None

    def test_force_recompute_flag(self, toy_dataset, simple_backbone):
        """Test force_recompute flag."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone=simple_backbone,
            precompute_embs=True,
            force_recompute=True
        )

        assert dm.force_recompute is True


class TestConceptDataModuleSplitter:
    """Test ConceptDataModule with custom splitters."""

    def test_custom_splitter(self, toy_dataset):
        """Test with custom splitter."""
        from torch_concepts.data.splitters.random import RandomSplitter

        splitter = RandomSplitter(val_size=0.15, test_size=0.15)

        dm = ConceptDataModule(
            dataset=toy_dataset,
            splitter=splitter
        )

        assert dm.splitter == splitter

        dm.setup('fit')

        # Check that splits are created
        assert dm.train_len > 0
        assert dm.val_len > 0
        assert dm.test_len > 0
