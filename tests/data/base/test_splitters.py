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


# ---------------------------------------------------------------------------
# CustomSplitter
# ---------------------------------------------------------------------------
from torch_concepts.data.splitters.custom import CustomSplitter


def _head_split(dataset, frac=0.2, mask=None):
    """Reference split fn: the first ``frac`` of the (optionally masked) pool
    is the split, the remainder is train. Returns ``(train_idxs, split_idxs)``.
    """
    n = len(dataset)
    masked = {int(i) for i in (mask or [])}
    pool = [i for i in range(n) if i not in masked]
    cut = int(frac * len(pool))
    return pool[cut:], pool[:cut]


class TestCustomSplitter:
    """Tests for CustomSplitter."""

    def test_basic_sizes_and_coverage(self):
        s = CustomSplitter(
            val_split_fn=_head_split, val_kwargs={"frac": 0.1},
            test_split_fn=_head_split, test_kwargs={"frac": 0.2},
        )
        s.fit(_LenDataset(100))

        assert s.test_len == 20            # 20% of 100
        assert s.val_len == 8              # 10% of the remaining 80
        assert s.train_len == 72
        tr, va, te = set(s.train_idxs), set(s.val_idxs), set(s.test_idxs)
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert len(tr | va | te) == 100    # full coverage, no overlap

    def test_test_indices_masked_out_of_val(self):
        received = {}

        def val_fn(dataset, frac=0.1, mask=None):
            received["mask"] = mask
            return _head_split(dataset, frac=frac, mask=mask)

        s = CustomSplitter(
            val_split_fn=val_fn, val_kwargs={"frac": 0.1},
            test_split_fn=_head_split, test_kwargs={"frac": 0.2},
            mask_test_indices_in_val=True,
        )
        s.fit(_LenDataset(100))

        assert received["mask"] is not None
        assert set(received["mask"]) == set(s.test_idxs)
        assert set(s.val_idxs).isdisjoint(s.test_idxs)

    def test_mask_not_passed_when_masking_disabled(self):
        received = {"has_mask": None}

        def val_fn(dataset, frac=0.1, **kwargs):
            received["has_mask"] = "mask" in kwargs
            return _head_split(dataset, frac=frac)

        s = CustomSplitter(
            val_split_fn=val_fn, val_kwargs={"frac": 0.1},
            test_split_fn=_head_split, test_kwargs={"frac": 0.2},
            mask_test_indices_in_val=False,
        )
        s.fit(_LenDataset(100))

        assert received["has_mask"] is False

    def test_kwargs_are_forwarded(self):
        received = {}

        def test_fn(dataset, frac=0.2, extra=None):
            received["frac"] = frac
            received["extra"] = extra
            return _head_split(dataset, frac=frac)

        s = CustomSplitter(
            test_split_fn=test_fn, test_kwargs={"frac": 0.3, "extra": "hi"},
        )
        s.fit(_LenDataset(100))

        assert received == {"frac": 0.3, "extra": "hi"}
        assert s.test_len == 30

    def test_kwargs_are_copied_not_shared(self):
        vk = {"frac": 0.1}
        s = CustomSplitter(
            val_split_fn=_head_split, val_kwargs=vk,
            test_split_fn=_head_split, test_kwargs={"frac": 0.2},
        )
        vk["frac"] = 0.99  # mutate the original after construction
        s.fit(_LenDataset(100))

        assert s.val_kwargs["frac"] == 0.1  # splitter unaffected
        assert s.val_len == 8

    def test_val_split_fn_none_puts_rest_in_train(self):
        s = CustomSplitter(test_split_fn=_head_split, test_kwargs={"frac": 0.2})
        s.fit(_LenDataset(100))

        assert s.test_len == 20
        assert s.val_len == 0
        assert s.train_len == 80
        assert set(s.train_idxs).isdisjoint(s.test_idxs)
        assert len(set(s.train_idxs) | set(s.test_idxs)) == 100

    def test_test_split_fn_none_gives_empty_test(self):
        s = CustomSplitter(val_split_fn=_head_split, val_kwargs={"frac": 0.1})
        s.fit(_LenDataset(100))

        assert s.test_idxs == []
        assert s.test_len == 0
        assert s.val_len == 10
        assert s.train_len == 90

    def test_both_fns_none_all_train(self):
        s = CustomSplitter()
        s.fit(_LenDataset(100))

        assert s.train_len == 100
        assert s.val_len == 0
        assert s.test_idxs == []

    def test_policies_reflect_function_names(self):
        s = CustomSplitter(val_split_fn=_head_split, test_split_fn=_head_split)
        assert s.val_policy == "_head_split"
        assert s.test_policy == "_head_split"

        empty = CustomSplitter()
        assert empty.val_policy is None
        assert empty.test_policy is None

    def test_fit_sets_fitted_flag(self):
        s = CustomSplitter(test_split_fn=_head_split)
        assert s.fitted is False
        s.fit(_LenDataset(100))
        assert s.fitted is True

    def test_split_is_cached_after_fit(self):
        calls = []

        def test_fn(dataset, **kwargs):
            calls.append(1)
            return _head_split(dataset, frac=0.2)

        s = CustomSplitter(test_split_fn=test_fn)
        s.split(_LenDataset(100))
        s.split(_LenDataset(100))  # already fitted -> must not recompute

        assert len(calls) == 1

    def test_repr_contains_policies_and_sizes(self):
        s = CustomSplitter(
            val_split_fn=_head_split, val_kwargs={"frac": 0.1},
            test_split_fn=_head_split, test_kwargs={"frac": 0.2},
        )
        s.fit(_LenDataset(100))
        r = repr(s)
        assert "CustomSplitter" in r
        assert "_head_split" in r
        assert "train_size=72" in r

    def test_integration_with_concept_datamodule(self, tmp_path):
        from torch_concepts.data.datasets.toy import ToyDataset
        from torch_concepts.data.base.datamodule import ConceptDataModule

        ds = ToyDataset("xor", n_gen=100, seed=0, root=str(tmp_path))
        dm = ConceptDataModule(
            ds,
            splitter=CustomSplitter(
                val_split_fn=_head_split, val_kwargs={"frac": 0.1},
                test_split_fn=_head_split, test_kwargs={"frac": 0.2},
            ),
        )
        dm.setup()

        assert dm.splitter.fitted
        assert dm.splitter.test_len == 20
        total = dm.splitter.train_len + dm.splitter.val_len + dm.splitter.test_len
        assert total == 100
        # CustomSplitter exposes no ``seed`` attribute; the datamodule must not
        # have forced one onto it.
        assert not hasattr(dm.splitter, "seed")


# ---------------------------------------------------------------------------
# FixedIndicesSplitter
# ---------------------------------------------------------------------------
from torch_concepts.data.splitters.fixed import FixedIndicesSplitter


class TestFixedIndicesSplitter:
    """Tests for FixedIndicesSplitter."""

    def test_indices_stored_as_given(self):
        s = FixedIndicesSplitter(
            train_idxs=[0, 1, 2, 3], val_idxs=[4, 5], test_idxs=[6, 7, 8]
        )
        assert s.train_idxs == [0, 1, 2, 3]
        assert s.val_idxs == [4, 5]
        assert s.test_idxs == [6, 7, 8]

    def test_lengths(self):
        s = FixedIndicesSplitter(
            train_idxs=range(70), val_idxs=range(70, 90), test_idxs=range(90, 100)
        )
        assert s.train_len == 70
        assert s.val_len == 20
        assert s.test_len == 10

    def test_fitted_immediately_after_construction(self):
        s = FixedIndicesSplitter(train_idxs=[0, 1], test_idxs=[2])
        assert s.fitted is True

    def test_fit_is_noop(self):
        s = FixedIndicesSplitter(
            train_idxs=[0, 1, 2], val_idxs=[3], test_idxs=[4]
        )
        before = (list(s.train_idxs), list(s.val_idxs), list(s.test_idxs))
        result = s.fit(_LenDataset(5))
        after = (list(s.train_idxs), list(s.val_idxs), list(s.test_idxs))
        assert result is None
        assert before == after

    def test_split_returns_indices_without_recompute(self):
        s = FixedIndicesSplitter(
            train_idxs=[0, 1, 2], val_idxs=[3], test_idxs=[4]
        )
        # Already fitted -> split() returns the stored index dict, untouched.
        indices = s.split(_LenDataset(5))
        assert indices["train"] == [0, 1, 2]
        assert indices["val"] == [3]
        assert indices["test"] == [4]

    def test_accepts_range_and_ndarray(self):
        s = FixedIndicesSplitter(
            train_idxs=range(3), val_idxs=np.array([3, 4]), test_idxs=[5]
        )
        # Normalized to plain lists.
        assert s.train_idxs == [0, 1, 2]
        assert s.val_idxs == [3, 4]
        assert isinstance(s.train_idxs, list) and isinstance(s.val_idxs, list)

    def test_indices_are_copied_not_shared(self):
        train = [0, 1, 2]
        s = FixedIndicesSplitter(train_idxs=train, test_idxs=[3])
        train.append(99)  # mutate the original after construction
        assert s.train_idxs == [0, 1, 2]

    def test_no_args_leaves_indices_none(self):
        s = FixedIndicesSplitter()
        assert s.fitted is True
        assert s.train_idxs is None
        assert s.val_idxs is None
        assert s.test_idxs is None
        assert s.train_len is None

    def test_partial_indices(self):
        s = FixedIndicesSplitter(train_idxs=[0, 1, 2], test_idxs=[3, 4])
        assert s.train_len == 3
        assert s.test_len == 2
        assert s.val_idxs is None
        assert s.val_len is None

    def test_repr_contains_sizes(self):
        s = FixedIndicesSplitter(
            train_idxs=range(70), val_idxs=range(20), test_idxs=range(10)
        )
        r = repr(s)
        assert "FixedIndicesSplitter" in r
        assert "train_size=70" in r
        assert "test_size=10" in r

    def test_integration_with_concept_datamodule(self, tmp_path):
        from torch_concepts.data.datasets.toy import ToyDataset
        from torch_concepts.data.base.datamodule import ConceptDataModule

        ds = ToyDataset("xor", n_gen=100, seed=0, root=str(tmp_path))
        train_idxs = list(range(0, 70))
        val_idxs = list(range(70, 90))
        test_idxs = list(range(90, 100))
        dm = ConceptDataModule(
            ds,
            splitter=FixedIndicesSplitter(
                train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs
            ),
        )
        dm.setup()

        assert list(dm.splitter.train_idxs) == train_idxs
        assert list(dm.splitter.val_idxs) == val_idxs
        assert list(dm.splitter.test_idxs) == test_idxs
        # No ``seed`` attribute should be forced onto it by the datamodule.
        assert not hasattr(dm.splitter, "seed")
