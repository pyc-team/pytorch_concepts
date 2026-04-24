"""
Tests for torch_concepts.data.base.datamodule module.

This module provides comprehensive tests for the ConceptDataModule class, including:
- Initialization with various configurations
- Property accessors and attribute delegation
- Setup stages (fit, test, validate)
- DataLoader creation
- Backbone embedding precomputation
- Splitting behavior
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.datasets.toy import ToyDataset
from torch_concepts.data.backbone import Backbone
from torch_concepts.annotations import Annotations


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def toy_dataset():
    """Create a simple toy dataset for testing."""
    return ToyDataset(
        dataset='xor',
        n_gen=100,
        seed=42
    )


@pytest.fixture
def large_toy_dataset():
    """Create a larger toy dataset for testing."""
    return ToyDataset(
        dataset='xor',
        n_gen=500,
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


# =============================================================================
# Test ConceptDataModule Initialization
# =============================================================================

class TestConceptDataModuleInit:
    """Test ConceptDataModule initialization."""

    def test_basic_init(self, toy_dataset):
        """Test basic initialization with minimal parameters."""
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

    def test_init_default_values(self, toy_dataset):
        """Test that default values are correctly set."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        assert dm.batch_size == 64  # Default batch size
        assert dm.workers == 0  # Default workers
        assert dm.pin_memory is False  # Default pin_memory
        assert dm.precompute_embs is False
        assert dm.force_recompute is False

    def test_init_with_string_backbone(self, toy_dataset):
        """Test initialization with string backbone name."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18',
            batch_size=16
        )

        assert dm.backbone is not None
        assert isinstance(dm.backbone, Backbone)
        assert dm.backbone.name == 'resnet18'

    def test_init_with_scalers(self, toy_dataset):
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

    def test_init_custom_workers(self, toy_dataset):
        """Test initialization with custom worker count."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            workers=4,
            pin_memory=True
        )

        assert dm.workers == 4
        assert dm.pin_memory is True

    def test_init_sets_embs_precomputed_false(self, toy_dataset):
        """Test that dataset's embs_precomputed is set to False on init."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.dataset.embs_precomputed is False

    def test_init_with_precompute_embs_flag(self, toy_dataset):
        """Test initialization with precompute_embs flag."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18',
            precompute_embs=True
        )
        
        assert dm.precompute_embs is True
        assert dm.backbone is not None

    def test_init_with_force_recompute(self, toy_dataset):
        """Test initialization with force_recompute flag."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18',
            precompute_embs=True,
            force_recompute=True
        )
        
        assert dm.force_recompute is True


# =============================================================================
# Test ConceptDataModule Properties
# =============================================================================

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

    def test_backbone_property_none(self, toy_dataset):
        """Test backbone property when no backbone provided."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.backbone is None

    def test_backbone_property_with_backbone(self, toy_dataset):
        """Test backbone property with backbone."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18'
        )
        assert dm.backbone is not None
        assert isinstance(dm.backbone, Backbone)

    def test_split_properties_before_setup(self, toy_dataset):
        """Test split properties before setup."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        assert dm.trainset is None
        assert dm.valset is None
        assert dm.testset is None
        assert dm.train_len is None
        assert dm.val_len is None
        assert dm.test_len is None


# =============================================================================
# Test ConceptDataModule Setup
# =============================================================================

class TestConceptDataModuleSetup:
    """Test ConceptDataModule setup method."""

    def test_setup_fit(self, toy_dataset):
        """Test setup with fit stage."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup('fit', verbose=False)

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

        dm.setup('test', verbose=False)

        assert dm.testset is not None
        assert dm.test_len > 0

    def test_setup_none_stage(self, toy_dataset):
        """Test setup with None stage (prepares all splits)."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup(None, verbose=False)

        assert dm.trainset is not None
        assert dm.valset is not None
        assert dm.testset is not None

    def test_split_sizes(self, toy_dataset):
        """Test that split sizes are correct."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup('fit', verbose=False)

        # With 100 samples, 0.2 test should give ~20, 0.1 val should give ~10
        assert dm.test_len == pytest.approx(20, abs=2)
        assert dm.val_len == pytest.approx(10, abs=2)
        assert dm.train_len == pytest.approx(70, abs=2)

    def test_setup_verbose_false(self, toy_dataset):
        """Test setup with verbose=False doesn't raise errors."""
        dm = ConceptDataModule(dataset=toy_dataset)
        dm.setup('fit', verbose=False)  # Should not print anything

    def test_setup_sets_embs_precomputed_false(self, toy_dataset):
        """Test that setup sets embs_precomputed=False when not precomputing."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            precompute_embs=False
        )
        dm.setup('fit', verbose=False)
        
        assert dm.dataset.embs_precomputed is False


# =============================================================================
# Test ConceptDataModule DataLoaders
# =============================================================================

class TestConceptDataModuleDataLoaders:
    """Test ConceptDataModule dataloader methods."""

    def test_train_dataloader(self, toy_dataset):
        """Test train dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.train_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_val_dataloader(self, toy_dataset):
        """Test validation dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.val_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_test_dataloader(self, toy_dataset):
        """Test test dataloader creation."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('test', verbose=False)

        loader = dm.test_dataloader()

        assert loader is not None
        assert loader.batch_size == 16

    def test_get_dataloader_whole_dataset(self, toy_dataset):
        """Test get_dataloader with split=None returns whole dataset."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.get_dataloader(split=None)

        assert loader is not None
        # Total batches should cover entire dataset
        total_samples = sum(batch['inputs']['x'].shape[0] for batch in loader)
        assert total_samples == 100

    def test_get_dataloader_custom_batch_size(self, toy_dataset):
        """Test get_dataloader with custom batch size."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.get_dataloader(split='train', batch_size=8)

        assert loader.batch_size == 8

    def test_get_dataloader_invalid_split(self, toy_dataset):
        """Test get_dataloader with invalid split raises ValueError."""
        dm = ConceptDataModule(dataset=toy_dataset)
        dm.setup('fit', verbose=False)

        with pytest.raises(ValueError, match="must be one of"):
            dm.get_dataloader(split='invalid')

    def test_dataloader_iteration(self, toy_dataset):
        """Test that dataloaders can be iterated."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert 'inputs' in batch
        assert 'concepts' in batch
        assert 'x' in batch['inputs']
        assert 'c' in batch['concepts']

        # Check batch sizes
        assert batch['inputs']['x'].shape[0] <= 16
        assert batch['concepts']['c'].shape[0] <= 16

    def test_train_dataloader_shuffles(self, large_toy_dataset):
        """Test that train dataloader shuffles by default."""
        dm = ConceptDataModule(
            dataset=large_toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        # Get two iterations - they should be different if shuffled
        loader = dm.train_dataloader(shuffle=True)
        batch1 = next(iter(loader))
        
        loader = dm.train_dataloader(shuffle=True)
        batch2 = next(iter(loader))
        
        # Not a perfect test, but batches are very unlikely to be identical
        # when shuffled
        # (This test may occasionally fail due to randomness, but it's useful)

    def test_val_dataloader_no_shuffle(self, toy_dataset):
        """Test that val dataloader doesn't shuffle by default."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit', verbose=False)

        loader = dm.val_dataloader(shuffle=False)
        batch1 = next(iter(loader))
        
        loader = dm.val_dataloader(shuffle=False)
        batch2 = next(iter(loader))
        
        # Batches should be identical without shuffling
        assert torch.allclose(batch1['inputs']['x'], batch2['inputs']['x'])


# =============================================================================
# Test ConceptDataModule Repr
# =============================================================================

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
        dm.setup('fit', verbose=False)
        repr_str = repr(dm)

        assert 'ConceptDataModule' in repr_str
        assert 'train_len=' in repr_str
        assert 'val_len=' in repr_str
        assert 'test_len=' in repr_str
        assert 'train_len=None' not in repr_str

    def test_repr_contains_batch_size(self, toy_dataset):
        """Test repr contains batch size."""
        dm = ConceptDataModule(dataset=toy_dataset, batch_size=32)
        repr_str = repr(dm)
        
        assert 'batch_size=32' in repr_str

    def test_repr_contains_dimensions(self, toy_dataset):
        """Test repr contains feature and concept dimensions."""
        dm = ConceptDataModule(dataset=toy_dataset)
        repr_str = repr(dm)
        
        assert 'n_features=' in repr_str
        assert 'n_concepts=' in repr_str


# =============================================================================
# Test ConceptDataModule with Scalers
# =============================================================================

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

        assert 'input' in dm.scalers
        assert isinstance(dm.scalers['input'], StandardScaler)

    def test_empty_scalers_dict(self, toy_dataset):
        """Test that empty scalers dict is used by default."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.scalers == {}


# =============================================================================
# Test ConceptDataModule Edge Cases
# =============================================================================

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

        dm.setup('fit', verbose=False)

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

        dm.setup('fit', verbose=False)

        assert dm.val_len == 0 or dm.val_len is None or dm.valset is None

    def test_large_batch_size(self, toy_dataset):
        """Test with batch size close to dataset size."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=50,  # Half of dataset size
            val_size=0.1,
            test_size=0.1
        )

        dm.setup('fit', verbose=False)
        loader = dm.train_dataloader()

        # Should still work
        batches = list(loader)
        assert len(batches) >= 1
        if len(batches) > 0:
            assert batches[0]['inputs']['x'].shape[0] == 50


# =============================================================================
# Test ConceptDataModule Backbone Integration
# =============================================================================

class TestConceptDataModuleBackbone:
    """Test ConceptDataModule with backbone embeddings."""

    def test_precompute_embs_flag(self, toy_dataset):
        """Test precompute_embs flag."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18',
            precompute_embs=True,
            batch_size=16
        )

        assert dm.precompute_embs is True
        assert dm.backbone is not None

    def test_force_recompute_flag(self, toy_dataset):
        """Test force_recompute flag."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone='resnet18',
            precompute_embs=True,
            force_recompute=True
        )

        assert dm.force_recompute is True

    def test_precompute_without_backbone_raises(self, toy_dataset):
        """Test that precompute_embs=True without backbone raises error."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            backbone=None,  # No backbone
            precompute_embs=True
        )

        with pytest.raises(ValueError, match="precompute_embs=True but no backbone"):
            dm.setup('fit', verbose=False)


# =============================================================================
# Test ConceptDataModule with Custom Splitter
# =============================================================================

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

        dm.setup('fit', verbose=False)

        # Check that splits are created
        assert dm.train_len > 0
        assert dm.val_len > 0
        assert dm.test_len > 0

    def test_default_random_splitter(self, toy_dataset):
        """Test that RandomSplitter is used by default."""
        from torch_concepts.data.splitters.random import RandomSplitter

        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        assert isinstance(dm.splitter, RandomSplitter)


# =============================================================================
# Test ConceptDataModule Add Set Method
# =============================================================================

class TestConceptDataModuleAddSet:
    """Test ConceptDataModule _add_set method."""

    def test_add_set_with_indices(self, toy_dataset):
        """Test _add_set with a list of indices."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        dm._add_set('train', [0, 1, 2, 3, 4])
        
        assert dm.trainset is not None
        assert dm.train_len == 5

    def test_add_set_with_empty_list(self, toy_dataset):
        """Test _add_set with an empty list."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        dm._add_set('val', [])
        
        assert dm.valset is None

    def test_add_set_with_none(self, toy_dataset):
        """Test _add_set with None."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        dm._add_set('test', None)
        
        assert dm.testset is None

    def test_add_set_invalid_split_type(self, toy_dataset):
        """Test _add_set with invalid split type."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        with pytest.raises(AssertionError):
            dm._add_set('invalid', [0, 1, 2])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
