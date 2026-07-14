"""Tests for the separation of the three seed roles in the data stack:

- **generation seed**: governs the sampled data (and its on-disk cache),
  exposed as ``seed`` on dataset classes and ``generation_seed`` on the
  generating datamodules. Kept fixed across runs.
- **split seed**: governs the train/val/test partition, exposed as ``seed``
  on the datamodules and routed to the splitter. The per-run knob.
- (the *train* seed — model init, shuffling, dropout — is the global seed and
  is exercised elsewhere; it is not a data-stack concern.)

These tests pin down that the two data-level seeds are independent and that
the split is reproducible regardless of whether data was freshly generated or
loaded from cache (the bug that motivated the dedicated split seed).
"""
import torch

from torch_concepts.data.datasets.toy import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.splitters.random import RandomSplitter
from torch_concepts.data.splitters.native import NativeSplitter
from torch_concepts.data.datamodules.completeness import CompletenessDataModule


def _toy(root, *, seed, n_gen=200):
    """Build a small toy dataset under an isolated root (per-seed cache)."""
    return ToyDataset(
        dataset="xor",
        n_gen=n_gen,
        seed=seed,
        root=str(root),
    )


# =============================================================================
# Generation seed
# =============================================================================

class TestGenerationSeed:
    def test_same_generation_seed_gives_identical_data(self, tmp_path):
        d1 = _toy(tmp_path / "a", seed=0)
        d2 = _toy(tmp_path / "b", seed=0)
        assert torch.equal(d1.input_data, d2.input_data)
        assert torch.equal(d1.concepts, d2.concepts)

    def test_different_generation_seed_gives_different_data(self, tmp_path):
        d1 = _toy(tmp_path / "a", seed=0)
        d2 = _toy(tmp_path / "b", seed=1)
        assert not torch.equal(d1.input_data, d2.input_data)

    def test_generation_seed_is_in_cache_filename(self, tmp_path):
        d0 = _toy(tmp_path / "a", seed=0)
        d1 = _toy(tmp_path / "b", seed=1)
        assert any("seed_0" in f for f in d0.processed_filenames)
        assert any("seed_1" in f for f in d1.processed_filenames)


# =============================================================================
# Split seed routing through the datamodule
# =============================================================================

class TestSplitSeedRouting:
    def test_seed_is_passed_to_default_splitter(self, tmp_path):
        dm = ConceptDataModule(_toy(tmp_path, seed=0), seed=5)
        assert isinstance(dm.splitter, RandomSplitter)
        assert dm.splitter.seed == 5

    def test_seed_propagated_to_passed_unseeded_splitter(self, tmp_path):
        splitter = RandomSplitter(val_size=0.1, test_size=0.2)  # seed=None
        dm = ConceptDataModule(_toy(tmp_path, seed=0), seed=5, splitter=splitter)
        assert dm.splitter.seed == 5

    def test_explicit_splitter_seed_is_not_overridden(self, tmp_path):
        splitter = RandomSplitter(seed=99)
        dm = ConceptDataModule(_toy(tmp_path, seed=0), seed=5, splitter=splitter)
        assert dm.splitter.seed == 99

    def test_seedless_splitter_type_is_left_untouched(self, tmp_path):
        # NativeSplitter has no ``seed`` attribute and must not gain one.
        dm = ConceptDataModule(
            _toy(tmp_path, seed=0), seed=5, splitter=NativeSplitter()
        )
        assert not hasattr(dm.splitter, "seed")

    def test_generating_datamodule_separates_the_two_seeds(self, tmp_path):
        """The config path: ``generation_seed`` -> dataset, ``seed`` -> split."""
        dm = CompletenessDataModule(
            name="completeness", root=str(tmp_path),
            seed=5, generation_seed=0,
            n_concepts=4, n_views=3, p=2, n_tasks=1, n_gen=200,
        )
        assert dm.dataset.seed == 0          # generation seed
        assert dm.splitter.seed == 5         # split seed


# =============================================================================
# Independence of split seed and generation seed
# =============================================================================

class TestSeedIndependence:
    def test_split_depends_only_on_split_seed_not_on_data(self, tmp_path):
        """Same split seed + different generated data -> identical split."""
        ds_a = _toy(tmp_path / "a", seed=0)
        ds_b = _toy(tmp_path / "b", seed=1)  # different data, same length

        sa = RandomSplitter(seed=5)
        sa.fit(ds_a)
        sb = RandomSplitter(seed=5)
        sb.fit(ds_b)

        assert sa.train_idxs == sb.train_idxs
        assert sa.test_idxs == sb.test_idxs

    def test_split_identical_whether_data_generated_or_cached(self, tmp_path):
        """The motivating bug: with a shared root, the first build generates
        data (consuming global RNG) and the second loads from cache. The split
        indices must be identical in both cases."""
        root = tmp_path / "shared"

        dm_gen = ConceptDataModule(_toy(root, seed=0), seed=5)  # cache miss
        dm_gen.setup()

        dm_load = ConceptDataModule(_toy(root, seed=0), seed=5)  # cache hit
        dm_load.setup()

        assert list(dm_gen.splitter.test_idxs) == list(dm_load.splitter.test_idxs)
        assert list(dm_gen.splitter.train_idxs) == list(dm_load.splitter.train_idxs)

    def test_different_split_seed_changes_split_same_data(self, tmp_path):
        ds = _toy(tmp_path, seed=0)
        s5 = RandomSplitter(seed=5)
        s5.fit(ds)
        s6 = RandomSplitter(seed=6)
        s6.fit(ds)
        assert s5.test_idxs != s6.test_idxs
