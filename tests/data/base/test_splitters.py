"""
Extended tests for torch_concepts.data.splitters to increase coverage.
"""
import pytest
import torch
import numpy as np


class TestRandomSplitterExtended:
    """Extended tests for RandomSplitter."""

    def test_random_splitter_fit_method(self):
        """Test RandomSplitter.fit() method with ConceptDataset."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        dataset = ToyDataset("xor", n_gen=100)
        splitter = RandomSplitter(val_size=0.2, test_size=0.1)

        # Fit should set train/val/test indices
        splitter.fit(dataset)

        assert hasattr(splitter, "train_idxs")
        assert hasattr(splitter, "val_idxs")
        assert hasattr(splitter, "test_idxs")

        # Check all indices are used exactly once
        all_indices = np.concatenate(
            [splitter.train_idxs, splitter.val_idxs, splitter.test_idxs]
        )
        assert len(all_indices) == 100
        assert len(np.unique(all_indices)) == 100

    def test_random_splitter_invalid_split_sizes(self):
        """Test RandomSplitter raises ValueError when splits exceed dataset size."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        dataset = ToyDataset("xor", n_gen=100)
        splitter = RandomSplitter(val_size=0.6, test_size=0.6)  # Sum > 1.0

        with pytest.raises(ValueError, match="Split sizes sum to"):
            splitter.fit(dataset)

    def test_random_splitter_fractional_sizes(self):
        """Test RandomSplitter with fractional split sizes."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        dataset = ToyDataset("xor", n_gen=100)
        splitter = RandomSplitter(val_size=0.15, test_size=0.25)

        splitter.fit(dataset)

        # Check approximate sizes (15% val, 25% test, 60% train)
        assert len(splitter.val_idxs) == 15
        assert len(splitter.test_idxs) == 25
        assert len(splitter.train_idxs) == 60

    def test_random_splitter_absolute_sizes(self):
        """Test RandomSplitter with absolute split sizes."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        dataset = ToyDataset("xor", n_gen=100)
        splitter = RandomSplitter(val_size=10, test_size=20)

        splitter.fit(dataset)

        assert len(splitter.val_idxs) == 10
        assert len(splitter.test_idxs) == 20
        assert len(splitter.train_idxs) == 70

    def test_random_splitter_no_validation(self):
        """Test RandomSplitter with zero validation size."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        dataset = ToyDataset("xor", n_gen=100)
        splitter = RandomSplitter(val_size=0, test_size=0.2)

        splitter.fit(dataset)

        assert len(splitter.val_idxs) == 0
        assert len(splitter.test_idxs) == 20
        assert len(splitter.train_idxs) == 80

    def test_random_splitter_basic(self):
        """Test RandomSplitter with basic settings using a dataset."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        splitter = RandomSplitter(val_size=0.2, test_size=0.1)

        dataset = ToyDataset("xor", n_gen=100)
        splitter.fit(dataset)

        # Check that all indices are used exactly once
        all_indices = np.concatenate([splitter.train_idxs, splitter.val_idxs, splitter.test_idxs])
        assert len(all_indices) == 100
        assert len(np.unique(all_indices)) == 100

    def test_random_splitter_no_test(self):
        """Test RandomSplitter with no test set."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        splitter = RandomSplitter(val_size=0.2, test_size=0.0)

        dataset = ToyDataset("xor", n_gen=100)
        splitter.fit(dataset)

        assert len(splitter.train_idxs) == 80
        assert len(splitter.val_idxs) == 20
        assert len(splitter.test_idxs) == 0

    def test_random_splitter_reproducible(self):
        """Test RandomSplitter reproducibility via its ``seed`` argument."""
        from torch_concepts.data.splitters.random import RandomSplitter
        from torch_concepts.data.datasets.toy import ToyDataset

        splitter1 = RandomSplitter(val_size=0.2, test_size=0.1, seed=42)
        dataset1 = ToyDataset("xor", n_gen=100)
        splitter1.fit(dataset1)
        train1 = splitter1.train_idxs
        val1 = splitter1.val_idxs
        test1 = splitter1.test_idxs

        # Same seed -> same split
        splitter2 = RandomSplitter(val_size=0.2, test_size=0.1, seed=42)
        dataset2 = ToyDataset("xor", n_gen=100)
        splitter2.fit(dataset2)
        train2 = splitter2.train_idxs
        val2 = splitter2.val_idxs
        test2 = splitter2.test_idxs

        assert np.array_equal(train1, train2)
        assert np.array_equal(val1, val2)
        assert np.array_equal(test1, test2)


class _LenDataset:
    """Minimal dataset stand-in exposing only ``__len__`` for the splitter."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class TestRandomSplitterSeed:
    """Tests for RandomSplitter's dedicated ``seed`` argument."""

    def test_seeded_split_is_reproducible(self):
        from torch_concepts.data.splitters.random import RandomSplitter

        s1 = RandomSplitter(val_size=0.2, test_size=0.1, seed=7)
        s1.fit(_LenDataset(100))
        s2 = RandomSplitter(val_size=0.2, test_size=0.1, seed=7)
        s2.fit(_LenDataset(100))

        assert s1.train_idxs == s2.train_idxs
        assert s1.val_idxs == s2.val_idxs
        assert s1.test_idxs == s2.test_idxs

    def test_seeded_split_independent_of_global_numpy_state(self):
        """A seeded split must not depend on whatever consumed numpy's
        global RNG beforehand (e.g. data generation)."""
        from torch_concepts.data.splitters.random import RandomSplitter

        np.random.seed(1)
        np.random.rand(123)  # perturb global state
        s1 = RandomSplitter(seed=7)
        s1.fit(_LenDataset(100))

        np.random.seed(2)
        np.random.rand(7)  # different global state
        s2 = RandomSplitter(seed=7)
        s2.fit(_LenDataset(100))

        assert s1.test_idxs == s2.test_idxs

    def test_different_seed_gives_different_split(self):
        from torch_concepts.data.splitters.random import RandomSplitter

        s1 = RandomSplitter(seed=7)
        s1.fit(_LenDataset(100))
        s2 = RandomSplitter(seed=8)
        s2.fit(_LenDataset(100))

        assert s1.test_idxs != s2.test_idxs

    def test_unseeded_split_ignores_global_numpy_and_is_nondeterministic(self):
        """Without a seed the split uses fresh entropy: it does NOT read the
        global numpy seed, and two unseeded splits differ."""
        from torch_concepts.data.splitters.random import RandomSplitter

        np.random.seed(42)
        s1 = RandomSplitter()
        s1.fit(_LenDataset(1000))
        np.random.seed(42)  # same global seed must NOT make the split repeat
        s2 = RandomSplitter()
        s2.fit(_LenDataset(1000))

        assert s1.seed is None
        assert s1.test_idxs != s2.test_idxs
