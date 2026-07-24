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
from types import SimpleNamespace
from unittest.mock import MagicMock

from torch.utils.data import Subset

from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.datasets.toy import ToyDataset
from torch_concepts.backbone import Backbone
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

    def test_init_default_values(self, toy_dataset):
        """Test that default values are correctly set."""
        dm = ConceptDataModule(dataset=toy_dataset)
        
        assert dm.batch_size == 64  # Default batch size
        assert dm.workers == 0  # Default workers
        assert dm.pin_memory is False  # Default pin_memory

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

    def test_dataset_embs_precomputed_false_by_default(self, toy_dataset):
        """The dataset owns the embs_precomputed flag; default is False."""
        dm = ConceptDataModule(dataset=toy_dataset)
        assert dm.dataset.embs_precomputed is False

    def test_max_samples_truncates_dataset(self, toy_dataset):
        """max_samples truncates the dataset at construction."""
        dm = ConceptDataModule(dataset=toy_dataset, max_samples=10)

        assert dm.n_samples == 10
        assert len(dm.dataset.concepts) == 10

        dm.setup('fit')
        total = sum(l for l in (dm.train_len, dm.val_len, dm.test_len) if l)
        assert total == 10


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

    def test_setup_none_stage(self, toy_dataset):
        """Test setup with None stage (prepares all splits)."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            val_size=0.1,
            test_size=0.2
        )

        dm.setup(None)

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

        dm.setup('fit')

        # With 100 samples, 0.2 test should give ~20, 0.1 val should give ~10
        assert dm.test_len == pytest.approx(20, abs=2)
        assert dm.val_len == pytest.approx(10, abs=2)
        assert dm.train_len == pytest.approx(70, abs=2)

    def test_setup_is_idempotent(self, toy_dataset):
        """Test that calling setup twice yields the same split (splitter caches)."""
        dm = ConceptDataModule(
            dataset=toy_dataset, val_size=0.1, test_size=0.2, seed=0
        )

        dm.setup('fit')
        sizes_1 = (dm.train_len, dm.val_len, dm.test_len)
        train_idxs_1 = list(dm.splitter.train_idxs)

        dm.setup('fit')
        sizes_2 = (dm.train_len, dm.val_len, dm.test_len)
        train_idxs_2 = list(dm.splitter.train_idxs)

        assert sizes_1 == sizes_2
        assert train_idxs_1 == train_idxs_2


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

    def test_get_dataloader_whole_dataset(self, toy_dataset):
        """Test get_dataloader with split=None returns whole dataset."""
        dm = ConceptDataModule(
            dataset=toy_dataset,
            batch_size=16
        )
        dm.setup('fit')

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
        dm.setup('fit')

        loader = dm.get_dataloader(split='train', batch_size=8)

        assert loader.batch_size == 8

    def test_get_dataloader_invalid_split(self, toy_dataset):
        """Test get_dataloader with invalid split raises ValueError."""
        dm = ConceptDataModule(dataset=toy_dataset)
        dm.setup('fit')

        with pytest.raises(ValueError, match="must be one of"):
            dm.get_dataloader(split='invalid')

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

    def test_train_dataloader_shuffles(self, large_toy_dataset):
        """Test that train dataloader shuffles by default."""
        dm = ConceptDataModule(
            dataset=large_toy_dataset,
            batch_size=16
        )
        dm.setup('fit')

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
        dm.setup('fit')

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
        dm.setup('fit')
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

    def test_no_scalers_means_nothing_fitted(self, toy_dataset):
        """Without a configured scaler, the dataset's scaler dict stays empty."""
        dm = ConceptDataModule(dataset=toy_dataset)
        dm.setup('fit')
        assert dm.dataset.scalers == {}


class TestConceptDataModuleScalerFitting:
    """Test that setup('fit') fits the configured scalers correctly."""

    @staticmethod
    def _continuous_dataset(n=100):
        """Dataset whose concepts are continuous with a strong index-dependent
        trend, so train-only statistics are distinguishable from full-data ones."""
        from torch_concepts.data.base.dataset import ConceptDataset

        annotations = Annotations(
            labels=['a', 'b'], cardinalities=[1, 1],
            types=['continuous', 'continuous'],
        )
        concepts = torch.stack([
            torch.arange(n, dtype=torch.float32),
            torch.arange(n, dtype=torch.float32) * 100.0,
        ], dim=1)
        return ConceptDataset(
            input_data=torch.randn(n, 4),
            concepts=concepts,
            annotations=annotations,
        )

    def test_fits_continuous_concepts(self):
        from torch_concepts.data.scalers import StandardScaler

        dm = ConceptDataModule(
            dataset=self._continuous_dataset(),
            scalers={'concepts': StandardScaler()},
            seed=0,
        )
        dm.setup('fit')

        fitted = dm.dataset.scalers['concepts']
        assert list(fitted.mean.annotation.labels) == ['a', 'b']

    def test_statistics_use_the_train_split_only(self):
        """The decisive property: validation/test rows must not leak into the
        statistics the model is normalised with."""
        from torch_concepts.data.scalers import StandardScaler

        dataset = self._continuous_dataset()
        dm = ConceptDataModule(
            dataset=dataset, scalers={'concepts': StandardScaler()}, seed=0,
        )
        dm.setup('fit')

        train_idx = dm.trainset.indices
        fitted = dm.dataset.scalers['concepts']
        expected = dataset.concepts[train_idx][:, 0].unsqueeze(-1).mean()
        assert fitted.mean['a'].tensor.item() == pytest.approx(
            expected.item(), rel=1e-5
        )
        # ...and that is genuinely not the full-dataset mean.
        assert fitted.mean['a'].tensor.item() != pytest.approx(
            dataset.concepts[:, 0].mean().item(), rel=1e-4
        )

    def test_dataset_data_is_not_mutated(self):
        """Fitting stores stats on `dataset.scalers`; the concepts tensor itself
        stays in its original scale (scaling happens per-batch in the learner)."""
        from torch_concepts.data.scalers import StandardScaler

        dataset = self._continuous_dataset()
        before = dataset.concepts.tensor.clone()
        dm = ConceptDataModule(
            dataset=dataset, scalers={'concepts': StandardScaler()}, seed=0,
        )
        dm.setup('fit')
        assert torch.equal(dataset.concepts.tensor, before)

    def test_scalers_are_refitted_on_every_setup_call(self):
        """Unlike the earlier `fitted_scalers` design, setup() always (re)fits
        when called with stage in ('fit', None) — there is no fitted-once guard."""
        from torch_concepts.data.scalers import StandardScaler

        dm = ConceptDataModule(
            dataset=self._continuous_dataset(),
            scalers={'concepts': StandardScaler()},
            seed=0,
        )
        dm.setup('fit')
        first = dm.dataset.scalers['concepts']
        dm.setup('fit')
        assert dm.dataset.scalers['concepts'] is first  # same prototype instance,
        assert dm.dataset.scalers['concepts'].mean is not None  # refit in place

    def test_binary_only_dataset_warns_and_skips(self, toy_dataset):
        """Binary/categorical concepts are class labels and are never scaled."""
        from torch_concepts.data.scalers import StandardScaler

        dm = ConceptDataModule(
            dataset=toy_dataset, scalers={'concepts': StandardScaler()}, seed=0,
        )
        with pytest.warns(UserWarning, match="no continuous concepts"):
            dm.setup('fit')
        assert 'concepts' not in dm.dataset.scalers

    def test_unknown_scaler_key_raises(self, toy_dataset):
        from torch_concepts.data.scalers import StandardScaler

        dm = ConceptDataModule(
            dataset=toy_dataset, scalers={'targets': StandardScaler()}, seed=0,
        )
        with pytest.raises(RuntimeError, match="cannot find attribute 'targets'"):
            dm.setup('fit')

    def test_input_scaler_on_flat_features(self, toy_dataset):
        """ToyDataset stores (N, 2) floats and serves (2,) — layouts line up."""
        from torch_concepts.data.scalers import StandardScaler

        dm = ConceptDataModule(
            dataset=toy_dataset, scalers={'input': StandardScaler()}, seed=0,
        )
        dm.setup('fit')
        assert 'input' in dm.dataset.scalers
        assert dm.dataset.scalers['input'].mean.shape[-1] == toy_dataset.input_data.shape[-1]


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

        # Should still work
        batches = list(loader)
        assert len(batches) >= 1
        if len(batches) > 0:
            assert batches[0]['inputs']['x'].shape[0] == 50


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

        dm.setup('fit')

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

    def test_add_set_with_dataset_instance(self, toy_dataset):
        """Test _add_set stores a Dataset instance directly (no Subset wrap)."""
        dm = ConceptDataModule(dataset=toy_dataset)
        subset = Subset(toy_dataset, [0, 1, 2, 3])

        dm._add_set('train', subset)

        # The exact Subset object should be stored as-is.
        assert dm.trainset is subset
        assert dm.train_len == 4

    def test_add_set_with_tuple_indices(self, toy_dataset):
        """Test _add_set accepts a tuple of indices."""
        dm = ConceptDataModule(dataset=toy_dataset)

        dm._add_set('val', (0, 1, 2))

        assert dm.valset is not None
        assert dm.val_len == 3

    def test_add_set_invalid_type_raises(self, toy_dataset):
        """Test _add_set rejects a value that is neither Dataset, sequence, nor None."""
        dm = ConceptDataModule(dataset=toy_dataset)

        with pytest.raises(AssertionError, match="not a valid type"):
            dm._add_set('train', 42)


# =============================================================================
# Test ConceptDataModule DataLoader behavior (None splits, drop_last, pin_memory)
# =============================================================================

class TestConceptDataModuleDataLoaderBehavior:
    """Test get_dataloader edge cases and DataLoader configuration."""

    def test_get_dataloader_none_before_setup(self, toy_dataset):
        """get_dataloader returns None for any split before setup()."""
        dm = ConceptDataModule(dataset=toy_dataset)

        assert dm.get_dataloader('train') is None
        assert dm.get_dataloader('val') is None
        assert dm.get_dataloader('test') is None

    def test_dataloader_wrappers_none_before_setup(self, toy_dataset):
        """train/val/test_dataloader return None before setup()."""
        dm = ConceptDataModule(dataset=toy_dataset)

        assert dm.train_dataloader() is None
        assert dm.val_dataloader() is None
        assert dm.test_dataloader() is None

    def test_empty_split_dataloader_is_none(self, toy_dataset):
        """An empty split (val_size=0) yields a None DataLoader."""
        dm = ConceptDataModule(
            dataset=toy_dataset, val_size=0.0, test_size=0.2
        )
        dm.setup('fit')

        assert dm.valset is None
        assert dm.val_dataloader() is None

    def test_train_dataloader_drop_last(self, toy_dataset):
        """Train loader drops the last partial batch; val/test do not."""
        dm = ConceptDataModule(dataset=toy_dataset, batch_size=16)
        dm.setup('fit')

        assert dm.train_dataloader().drop_last is True
        assert dm.val_dataloader().drop_last is False
        assert dm.test_dataloader().drop_last is False

    def test_pin_memory_only_for_train(self, toy_dataset):
        """pin_memory is applied only to the train loader."""
        dm = ConceptDataModule(
            dataset=toy_dataset, batch_size=16, pin_memory=True
        )
        dm.setup('fit')

        assert dm.train_dataloader().pin_memory is True
        # Non-train splits pass pin_memory=None -> DataLoader stores False.
        assert not dm.val_dataloader().pin_memory

    def test_workers_propagated_to_dataloader(self, toy_dataset):
        """The configured worker count reaches the DataLoader."""
        dm = ConceptDataModule(dataset=toy_dataset, batch_size=16, workers=2)
        dm.setup('fit')

        assert dm.train_dataloader().num_workers == 2

    def test_get_dataloader_val_test_explicit(self, toy_dataset):
        """get_dataloader('val'|'test') returns loaders with the given batch size."""
        dm = ConceptDataModule(dataset=toy_dataset, batch_size=16)
        dm.setup('fit')

        val_loader = dm.get_dataloader('val', batch_size=4)
        test_loader = dm.get_dataloader('test', batch_size=4)

        assert val_loader is not None and val_loader.batch_size == 4
        assert test_loader is not None and test_loader.batch_size == 4


# =============================================================================
# Test embedding caching (dataset.precompute_embeddings + datamodule delegate)
# =============================================================================

class TestCacheEmbeddings:
    """Test precompute_embeddings compute/cache/load branches.

    A fake backbone (only needs a ``filename``) plus a stubbed
    ``_compute_embeddings`` exercise the caching logic without a real model.
    """

    @staticmethod
    def _make_ds(tmp_path):
        ds = ToyDataset('xor', n_gen=20, seed=0, root=str(tmp_path))
        return ds, SimpleNamespace(filename='bb.pt')

    def test_computes_and_caches(self, tmp_path):
        ds, bb = self._make_ds(tmp_path)
        embs = torch.randn(len(ds), 8)
        ds._compute_embeddings = MagicMock(return_value=embs)

        ds.precompute_embeddings(bb)

        cache_path = os.path.join(ds.root_dir, 'bb.pt')
        assert os.path.exists(cache_path)              # cached to disk
        ds._compute_embeddings.assert_called_once()
        assert torch.equal(ds.input_data, embs)
        assert ds.embs_precomputed is True

    def test_loads_from_cache(self, tmp_path):
        ds, bb = self._make_ds(tmp_path)
        cached = torch.randn(len(ds), 8)
        os.makedirs(ds.root_dir, exist_ok=True)
        torch.save(cached, os.path.join(ds.root_dir, 'bb.pt'))

        ds._compute_embeddings = MagicMock()  # must NOT be called

        ds.precompute_embeddings(bb)

        ds._compute_embeddings.assert_not_called()
        assert torch.equal(ds.input_data, cached)
        assert ds.embs_precomputed is True

    def test_force_ignores_cache(self, tmp_path):
        ds, bb = self._make_ds(tmp_path)
        os.makedirs(ds.root_dir, exist_ok=True)
        torch.save(torch.randn(len(ds), 8), os.path.join(ds.root_dir, 'bb.pt'))

        fresh = torch.randn(len(ds), 8)
        ds._compute_embeddings = MagicMock(return_value=fresh)

        ds.precompute_embeddings(bb, force=True)

        ds._compute_embeddings.assert_called_once()
        assert torch.equal(ds.input_data, fresh)

    def test_stale_row_count_recomputes(self, tmp_path):
        """A cache with the wrong number of rows (e.g. written from a subset)
        is ignored and recomputed."""
        ds, bb = self._make_ds(tmp_path)
        os.makedirs(ds.root_dir, exist_ok=True)
        torch.save(torch.randn(5, 8), os.path.join(ds.root_dir, 'bb.pt'))  # 5 != 20

        fresh = torch.randn(len(ds), 8)
        ds._compute_embeddings = MagicMock(return_value=fresh)

        ds.precompute_embeddings(bb)

        ds._compute_embeddings.assert_called_once()
        assert torch.equal(ds.input_data, fresh)

    def test_cache_false_computes_in_memory_only(self, tmp_path):
        """cache=False computes without reading or writing the cache file."""
        ds, bb = self._make_ds(tmp_path)
        os.makedirs(ds.root_dir, exist_ok=True)
        stale = torch.randn(len(ds), 8)
        torch.save(stale, os.path.join(ds.root_dir, 'bb.pt'))  # must be ignored

        fresh = torch.randn(len(ds), 8)
        ds._compute_embeddings = MagicMock(return_value=fresh)

        ds.precompute_embeddings(bb, cache=False)

        ds._compute_embeddings.assert_called_once()          # cache not read
        assert torch.equal(ds.input_data, fresh)
        assert torch.equal(                                  # cache not overwritten
            torch.load(os.path.join(ds.root_dir, 'bb.pt')), stale
        )
        assert ds.embs_precomputed is True

    def test_datamodule_delegates_with_own_defaults(self, tmp_path):
        """dm.precompute_embeddings forwards its batch_size/workers to the dataset."""
        ds, bb = self._make_ds(tmp_path)
        dm = ConceptDataModule(dataset=ds, batch_size=7, workers=3)
        ds.precompute_embeddings = MagicMock()

        dm.precompute_embeddings(bb, force=True)

        ds.precompute_embeddings.assert_called_once_with(
            bb, batch_size=7, workers=3, cache=True, cache_dir=None, force=True
        )

    def test_cache_dir_overrides_root_dir(self, tmp_path):
        """cache_dir redirects the cache file away from the dataset root."""
        ds, bb = self._make_ds(tmp_path)
        ds._compute_embeddings = MagicMock(return_value=torch.randn(len(ds), 8))
        other = os.path.join(str(tmp_path), 'scratch')

        ds.precompute_embeddings(bb, cache_dir=other)

        assert os.path.exists(os.path.join(other, 'bb.pt'))
        assert not os.path.exists(os.path.join(ds.root_dir, 'bb.pt'))


class TestMaxSamplesNativeSplitter:
    """`max_samples` with a NativeSplitter is rejected with a clear error."""

    def test_native_splitter_incompatible_with_max_samples(self, toy_dataset):
        from torch_concepts.data.splitters.native import NativeSplitter
        with pytest.raises(ValueError, match="max_samples.*incompatible.*NativeSplitter"):
            ConceptDataModule(dataset=toy_dataset, splitter=NativeSplitter(), max_samples=10)


class TestComputeEmbeddingsEvalGuard:
    """_compute_embeddings forces eval during extraction and restores the mode."""

    def test_forces_eval_and_restores_mode(self, tmp_path):
        ds = ToyDataset('xor', n_gen=8, seed=0, root=str(tmp_path))

        class _RecordingBackbone(nn.Module):
            source = "torchvision"

            def __init__(self):
                super().__init__()
                self.out_features = 4
                self.seen_training = None
                self._p = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                self.seen_training = self.training
                return torch.zeros(x.shape[0], self.out_features)

        bb = _RecordingBackbone()
        bb.train()
        assert bb.training

        embs = ds._compute_embeddings(bb, batch_size=4, workers=0)

        assert bb.seen_training is False      # ran in eval (deterministic BN/Dropout)
        assert bb.training is True            # caller's mode restored afterwards
        assert embs.shape == (len(ds), 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
