#!/usr/bin/env python3
"""
Tests for ToyDataset and CompletenessDataset classes.

This module tests the implementation of toy datasets including XOR, Trigonometry,
Dot, Checkmark, and the CompletenessDataset.
"""
import pytest
import tempfile
import shutil
import os
import numpy as np
import torch
import pandas as pd
from torch_concepts.data.datasets.toy import ToyDataset, CompletenessDataset, TOYDATASETS


class TestToyDataset:
    """Test suite for ToyDataset class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("dataset_name", TOYDATASETS)
    def test_toy_dataset_creation(self, temp_dir, dataset_name):
        """Test that each toy dataset can be created successfully."""
        dataset = ToyDataset(
            dataset=dataset_name,
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        assert dataset is not None
        assert len(dataset) == 100
        assert dataset.dataset_name == dataset_name.lower()

    @pytest.mark.parametrize("dataset_name", TOYDATASETS)
    def test_toy_dataset_properties(self, temp_dir, dataset_name):
        """Test that dataset properties are correctly set."""
        dataset = ToyDataset(
            dataset=dataset_name,
            root=temp_dir,
            seed=42,
            n_gen=200
        )

        # Check basic properties (n_features might be a tuple)
        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features > 0
        assert dataset.n_concepts > 0
        assert len(dataset.concept_names) == dataset.n_concepts

        # Check that annotations exist
        assert dataset.annotations is not None
        assert dataset.annotations.labels is not None

    def test_xor_dataset_structure(self, temp_dir):
        """Test XOR dataset specific structure."""
        dataset = ToyDataset(
            dataset='xor',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features == 2
        assert dataset.n_concepts == 3  # C1, C2, xor (includes task)
        assert dataset.concept_names == ['C1', 'C2', 'xor']

        # Check sample structure
        sample = dataset[0]
        assert 'inputs' in sample
        assert 'concepts' in sample
        assert sample['inputs']['x'].shape == (2,)
        assert sample['concepts']['c'].shape == (3,)  # includes task

    def test_trigonometry_dataset_structure(self, temp_dir):
        """Test Trigonometry dataset specific structure."""
        dataset = ToyDataset(
            dataset='trigonometry',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features == 7
        assert dataset.n_concepts == 4  # C1, C2, C3, sumGreaterThan1 (includes task)
        assert dataset.concept_names == ['C1', 'C2', 'C3', 'sumGreaterThan1']

        # Check sample structure
        sample = dataset[0]
        assert sample['inputs']['x'].shape == (7,)
        assert sample['concepts']['c'].shape == (4,)  # includes task

    def test_dot_dataset_structure(self, temp_dir):
        """Test Dot dataset specific structure."""
        dataset = ToyDataset(
            dataset='dot',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features == 4
        assert dataset.n_concepts == 3  # dotV1V2GreaterThan0, dotV3V4GreaterThan0, dotV1V3GreaterThan0 (includes task)
        assert dataset.concept_names == ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0', 'dotV1V3GreaterThan0']

        # Check sample structure
        sample = dataset[0]
        assert sample['inputs']['x'].shape == (4,)
        assert sample['concepts']['c'].shape == (3,)  # includes task

    def test_checkmark_dataset_structure(self, temp_dir):
        """Test Checkmark dataset specific structure."""
        dataset = ToyDataset(
            dataset='checkmark',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features == 4
        assert dataset.n_concepts == 4  # A, B, C, D (includes task)
        assert dataset.concept_names == ['A', 'B', 'C', 'D']

        # Check that graph exists for checkmark
        assert dataset.graph is not None

        # Check sample structure
        sample = dataset[0]
        assert sample['inputs']['x'].shape == (4,)
        assert sample['concepts']['c'].shape == (4,)  # includes task

    def test_toy_dataset_reproducibility(self, temp_dir):
        """Test that datasets are reproducible with the same seed."""
        dataset1 = ToyDataset(
            dataset='xor',
            root=os.path.join(temp_dir, 'ds1'),
            seed=42,
            n_gen=50
        )

        dataset2 = ToyDataset(
            dataset='xor',
            root=os.path.join(temp_dir, 'ds2'),
            seed=42,
            n_gen=50
        )

        # Check that data is identical
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])
        assert torch.allclose(sample1['concepts']['c'], sample2['concepts']['c'])

    def test_toy_dataset_different_seeds(self, temp_dir):
        """Test that different seeds produce different data."""
        dataset1 = ToyDataset(
            dataset='xor',
            root=os.path.join(temp_dir, 'ds1'),
            seed=42,
            n_gen=50
        )

        dataset2 = ToyDataset(
            dataset='xor',
            root=os.path.join(temp_dir, 'ds2'),
            seed=123,
            n_gen=50
        )

        # Check that data is different
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert not torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])

    def test_toy_dataset_persistence(self, temp_dir):
        """Test that dataset is saved and can be loaded."""
        # Create dataset
        dataset1 = ToyDataset(
            dataset='xor',
            root=temp_dir,
            seed=42,
            n_gen=50
        )
        sample1 = dataset1[0]

        # Load the same dataset again (should load from disk)
        dataset2 = ToyDataset(
            dataset='xor',
            root=temp_dir,
            seed=42,
            n_gen=50
        )
        sample2 = dataset2[0]

        # Check that data is identical
        assert torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])
        assert torch.allclose(sample1['concepts']['c'], sample2['concepts']['c'])

    def test_toy_dataset_invalid_name(self, temp_dir):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="Dataset .* not found"):
            ToyDataset(
                dataset='invalid_dataset',
                root=temp_dir,
                seed=42,
                n_gen=100
            )

    def test_toy_dataset_concept_subset(self, temp_dir):
        """Test that concept subset selection works."""
        dataset = ToyDataset(
            dataset='trigonometry',
            root=temp_dir,
            seed=42,
            n_gen=100,
            concept_subset=['C1', 'C2']
        )

        # Should only have 2 concepts selected
        assert dataset.n_concepts == 2
        assert 'C1' in dataset.concept_names
        assert 'C2' in dataset.concept_names
        assert 'C3' not in dataset.concept_names

    def test_toy_dataset_annotations_metadata(self, temp_dir):
        """Test that annotations contain proper metadata."""
        dataset = ToyDataset(
            dataset='xor',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        # Check annotations structure
        assert dataset.annotations.cardinalities is not None
        assert dataset.annotations.types is not None

        # All concepts should be discrete (binary/categorical, not continuous)
        for concept_name in dataset.concept_names:
            assert dataset.annotations.concept(concept_name).type in ('binary', 'categorical')

    def test_toy_dataset_batching(self, temp_dir):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ToyDataset(
            dataset='xor',
            root=temp_dir,
            seed=42,
            n_gen=100
        )

        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        batch = next(iter(dataloader))

        assert batch['inputs']['x'].shape == (10, 2)
        assert batch['concepts']['c'].shape == (10, 3)  # includes task (C1, C2, xor)


class TestCompletenessDataset:
    """Test suite for CompletenessDataset class."""

    def test_structure(self):
        from torch.utils.data import DataLoader

        ds = CompletenessDataset(n_gen=100, n_concepts=10, observed_fraction=0.7,
                                 hidden_importance=0.3, n_side=3)
        assert ds.concept_names == [f"C{i}" for i in range(7)] + ["y"]
        assert ds.input_data.shape == (100, 13)
        assert ds.hidden_concepts.shape == (100, 3)
        assert ds.side_latents.shape == (100, 3)
        batch = next(iter(DataLoader(ds, batch_size=10)))
        assert batch['inputs']['x'].shape == (10, 13)
        assert batch['concepts']['c'].shape == (10, 8)

    def test_seed(self):
        a, b, c = (CompletenessDataset(n_gen=50, seed=s) for s in (0, 0, 1))
        assert torch.equal(a.input_data, b.input_data)
        assert torch.equal(a.concepts.tensor, b.concepts.tensor)
        assert not torch.equal(a.input_data, c.input_data)

    @pytest.mark.parametrize("n_concepts,nonlinearity", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (10, 0.5)])
    def test_balanced(self, n_concepts, nonlinearity):
        # the previous generator drew a constant task for <= 3 concepts
        c = CompletenessDataset(n_gen=20000, n_concepts=n_concepts, nonlinearity=nonlinearity).concepts.tensor
        assert ((c.mean(0) - 0.5).abs() < 0.02).all()

    def test_y_is_a_function_of_x(self):
        ds = CompletenessDataset(n_gen=20000, n_concepts=6, observed_fraction=0.5, hidden_importance=0.5,
                                 nonlinearity=0.5, side_importance=0.3, mixing_layers=4)
        v = ds.input_data.double().numpy() @ ds._mix_w[-1].T  # invert the deepest mixing network
        for w, b in zip(reversed(ds._mix_w[:-1]), reversed(ds._mix_b)):
            v = (np.where(v > 0, v, v / 0.2) / np.sqrt(2 / 1.04) - b) @ w.T
        y = ds._label(ds._score(v[:, :6], v[:, 6:10]))
        assert np.array_equal(y, ds.concepts.tensor[:, -1].numpy())  # exact, despite float32 x
        assert np.array_equal(v[:, 3:6] > 0, ds.hidden_concepts.numpy() == 1)

    def test_hidden_observed_corr(self):
        ds = CompletenessDataset(n_gen=10, n_concepts=4, observed_fraction=0.5, hidden_observed_corr=-0.6)
        z = ds._latents(np.random.default_rng(0), 100_000)[0]
        c = z > 0
        assert abs(np.corrcoef(c[:, 0], c[:, 2])[0, 1] + 0.6) < 0.01  # C2, hidden, is paired with C0
        on = c[:, 0]  # given C0, its latent (a predictor's confidence) says nothing about C2
        assert abs(np.corrcoef(z[on, 0], c[on, 2])[0, 1]) < 0.02

    def test_bayes_accuracy_follows_the_knobs(self):
        def acc(given="observed", **kw):
            ds = CompletenessDataset(n_gen=10, n_concepts=8, observed_fraction=0.5, **kw)
            return ds.bayes_accuracy(given, n=1000, n_draws=100)

        assert acc(hidden_importance=0.0) == 1.0
        assert acc(hidden_importance=1.0) < 0.6
        assert acc(hidden_importance=0.5, hidden_observed_corr=1.0) == 1.0  # hidden concepts copy observed ones
        assert acc("all", hidden_importance=0.5) == 1.0
        assert acc("all", side_importance=0.5) < 0.9
        accs = [acc(hidden_importance=h) for h in (0.1, 0.4, 0.8)]
        assert accs[0] > accs[1] > accs[2]

    def test_nonlinearity(self):
        from sklearn.linear_model import LogisticRegression

        def linear_head_accuracy(nonlinearity):
            c = CompletenessDataset(n_gen=4000, n_concepts=6, nonlinearity=nonlinearity).concepts.tensor.numpy()
            return LogisticRegression(C=1e4, max_iter=1000).fit(c[:, :-1], c[:, -1]).score(c[:, :-1], c[:, -1])

        assert linear_head_accuracy(0.0) > 0.99
        assert linear_head_accuracy(1.0) < 0.8

    def test_knobs_change_only_y(self):
        base = dict(n_gen=500, n_concepts=8, observed_fraction=0.5, hidden_importance=0.5)
        ref = CompletenessDataset(**base)
        for knob in [dict(hidden_importance=0.2), dict(nonlinearity=0.7), dict(side_importance=0.3)]:
            ds = CompletenessDataset(**{**base, **knob})
            assert torch.equal(ds.input_data, ref.input_data)
            assert torch.equal(ds.concepts.tensor[:, :-1], ref.concepts.tensor[:, :-1])
            assert torch.equal(ds.hidden_concepts, ref.hidden_concepts)
        # with hidden_importance=None, observed_fraction only masks a fixed dataset
        full = CompletenessDataset(n_gen=500, n_concepts=8)
        part = CompletenessDataset(n_gen=500, n_concepts=8, observed_fraction=0.5)
        c = part.concepts.tensor
        assert torch.equal(full.input_data, part.input_data)
        assert torch.equal(full.concepts.tensor, torch.cat([c[:, :-1], part.hidden_concepts, c[:, -1:]], 1))

    def test_graph(self):
        def edges(**kw):
            g = CompletenessDataset(n_gen=50, n_concepts=4, **kw).graph
            return {(s, t) for s in g.node_names for t in g.get_successors(s)}

        assert edges() == {(f"C{i}", "y") for i in range(4)}
        assert edges(observed_fraction=0.5, hidden_importance=1.0) == set()

    def test_concept_subset(self):
        ds = CompletenessDataset(n_gen=50, n_concepts=5, concept_subset=['C0', 'C3', 'y'])
        assert ds.concept_names == ['C0', 'C3', 'y']

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            CompletenessDataset(n_concepts=4, hidden_importance=0.5)  # nothing is hidden
        with pytest.raises(ValueError):
            CompletenessDataset(nonlinearity=1.5)
        with pytest.raises(ValueError):
            CompletenessDataset(mixing_layers=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
