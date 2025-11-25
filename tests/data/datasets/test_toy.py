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
        assert 1 in dataset.annotations
        assert dataset.annotations[1].labels is not None

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
        assert dataset.annotations[1].cardinalities is not None
        assert dataset.annotations[1].metadata is not None

        # All concepts should be discrete
        for concept_name in dataset.concept_names:
            assert dataset.annotations[1].metadata[concept_name]['type'] == 'discrete'

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

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_completeness_dataset_creation(self, temp_dir):
        """Test that completeness dataset can be created."""
        dataset = CompletenessDataset(
            name='test_completeness',
            root=temp_dir,
            seed=42,
            n_gen=100,
            n_concepts=3,
            n_hidden_concepts=0
        )

        assert dataset is not None
        assert len(dataset) == 100
        assert dataset.name == 'test_completeness'

    def test_completeness_dataset_properties(self, temp_dir):
        """Test that completeness dataset properties are correct."""
        n_concepts = 5
        n_gen = 200

        dataset = CompletenessDataset(
            name='test_complete',
            root=temp_dir,
            seed=42,
            n_gen=n_gen,
            n_concepts=n_concepts,
            n_hidden_concepts=0
        )

        assert len(dataset) == n_gen
        assert dataset.n_concepts == n_concepts + 1  # includes task
        assert len(dataset.concept_names) == n_concepts + 1

        # Check concept names format - should be C0, C1, ..., y0
        for i in range(n_concepts):
            assert f'C{i}' in dataset.concept_names
        assert 'y0' in dataset.concept_names

    def test_completeness_dataset_with_hidden_concepts(self, temp_dir):
        """Test completeness dataset with hidden concepts."""
        dataset = CompletenessDataset(
            name='test_hidden',
            root=temp_dir,
            seed=42,
            n_gen=100,
            n_concepts=3,
            n_hidden_concepts=2
        )

        # Should expose n_concepts + n_tasks (3 concepts + 1 task = 4)
        assert dataset.n_concepts == 4  # 3 concepts + 1 task
        assert len(dataset.concept_names) == 4

    def test_completeness_dataset_structure(self, temp_dir):
        """Test completeness dataset structure."""
        p = 2
        n_views = 10
        n_concepts = 4

        dataset = CompletenessDataset(
            name='test_structure',
            root=temp_dir,
            seed=42,
            n_gen=50,
            p=p,
            n_views=n_views,
            n_concepts=n_concepts
        )

        # Input features should be p * n_views
        expected_features = p * n_views
        n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
        assert n_features == expected_features

        # Check sample structure - includes task
        sample = dataset[0]
        assert 'inputs' in sample
        assert 'concepts' in sample
        assert sample['inputs']['x'].shape == (expected_features,)
        assert sample['concepts']['c'].shape == (n_concepts + 1,)  # includes task

    def test_completeness_dataset_reproducibility(self, temp_dir):
        """Test that completeness dataset is reproducible with same seed."""
        dataset1 = CompletenessDataset(
            name='test_repro1',
            root=os.path.join(temp_dir, 'ds1'),
            seed=42,
            n_gen=50,
            n_concepts=3
        )

        dataset2 = CompletenessDataset(
            name='test_repro2',
            root=os.path.join(temp_dir, 'ds2'),
            seed=42,
            n_gen=50,
            n_concepts=3
        )

        # Check that data is identical
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])
        assert torch.allclose(sample1['concepts']['c'], sample2['concepts']['c'])

    def test_completeness_dataset_different_seeds(self, temp_dir):
        """Test that different seeds produce different data."""
        dataset1 = CompletenessDataset(
            name='test_seed1',
            root=os.path.join(temp_dir, 'ds1'),
            seed=42,
            n_gen=50,
            n_concepts=3
        )

        dataset2 = CompletenessDataset(
            name='test_seed2',
            root=os.path.join(temp_dir, 'ds2'),
            seed=123,
            n_gen=50,
            n_concepts=3
        )

        # Check that data is different
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert not torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])

    def test_completeness_dataset_persistence(self, temp_dir):
        """Test that completeness dataset is saved and loaded correctly."""
        # Create dataset
        dataset1 = CompletenessDataset(
            name='test_persist',
            root=temp_dir,
            seed=42,
            n_gen=50,
            n_concepts=3
        )
        sample1 = dataset1[0]

        # Load the same dataset again (should load from disk)
        dataset2 = CompletenessDataset(
            name='test_persist',
            root=temp_dir,
            seed=42,
            n_gen=50,
            n_concepts=3
        )
        sample2 = dataset2[0]

        # Check that data is identical
        assert torch.allclose(sample1['inputs']['x'], sample2['inputs']['x'])
        assert torch.allclose(sample1['concepts']['c'], sample2['concepts']['c'])

    def test_completeness_dataset_no_graph(self, temp_dir):
        """Test that completeness dataset has a graph."""
        dataset = CompletenessDataset(
            name='test_graph',
            root=temp_dir,
            seed=42,
            n_gen=50,
            n_concepts=3
        )

        # Completeness datasets should have a graph
        assert dataset.graph is not None

    def test_completeness_dataset_concept_subset(self, temp_dir):
        """Test that concept subset selection works."""
        dataset = CompletenessDataset(
            name='test_subset',
            root=temp_dir,
            seed=42,
            n_gen=100,
            n_concepts=5,
            concept_subset=['C0', 'C1', 'C3']
        )

        # Should only have 3 concepts selected
        assert dataset.n_concepts == 3
        assert 'C0' in dataset.concept_names
        assert 'C1' in dataset.concept_names
        assert 'C3' in dataset.concept_names
        assert 'C2' not in dataset.concept_names
        assert 'C4' not in dataset.concept_names

    def test_completeness_dataset_annotations(self, temp_dir):
        """Test that completeness dataset annotations are correct."""
        dataset = CompletenessDataset(
            name='test_annotations',
            root=temp_dir,
            seed=42,
            n_gen=100,
            n_concepts=3
        )

        # Check annotations structure
        assert dataset.annotations is not None
        assert 1 in dataset.annotations
        assert dataset.annotations[1].labels is not None
        assert dataset.annotations[1].cardinalities is not None
        assert dataset.annotations[1].metadata is not None

        # All concepts should be discrete
        for concept_name in dataset.concept_names:
            assert dataset.annotations[1].metadata[concept_name]['type'] == 'discrete'

    def test_completeness_dataset_batching(self, temp_dir):
        """Test that completeness dataset works with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = CompletenessDataset(
            name='test_batching',
            root=temp_dir,
            seed=42,
            n_gen=100,
            p=2,
            n_views=5,
            n_concepts=3
        )

        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        batch = next(iter(dataloader))

        assert batch['inputs']['x'].shape == (10, 10)  # 10 samples, 2*5 features
        assert batch['concepts']['c'].shape == (10, 4)  # 10 samples, 3 concepts + 1 task

    def test_completeness_dataset_different_parameters(self, temp_dir):
        """Test completeness dataset with various parameter combinations."""
        params_list = [
            {'p': 2, 'n_views': 5, 'n_concepts': 2},
            {'p': 3, 'n_views': 7, 'n_concepts': 4},
            {'p': 1, 'n_views': 10, 'n_concepts': 3},
        ]

        for i, params in enumerate(params_list):
            dataset = CompletenessDataset(
                name=f'test_params_{i}',
                root=os.path.join(temp_dir, f'ds_{i}'),
                seed=42,
                n_gen=50,
                **params
            )

            n_features = dataset.n_features[0] if isinstance(dataset.n_features, tuple) else dataset.n_features
            assert n_features == params['p'] * params['n_views']
            assert dataset.n_concepts == params['n_concepts'] + 1  # includes task


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
