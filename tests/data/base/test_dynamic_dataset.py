"""Tests for DynamicConceptDataset: windows of time series as samples."""
import pytest
import torch

from torch_concepts.annotations import Annotations
from torch_concepts.data.base import ConceptDataModule, DynamicConceptDataset
from torch_concepts.data.splitters import FixedIndicesSplitter

ANNOTATIONS = Annotations(labels=["c0", "c1"], cardinalities=[1, 1], types=["continuous"] * 2)


def series(n_steps=30, n_series=4):
    """Inputs (T, N, 3) and concepts (T, N, 2) whose values are 100 * step + series."""
    values = 100 * torch.arange(n_steps)[:, None] + torch.arange(n_series)[None]
    values = values.float()[..., None]
    return values.expand(-1, -1, 3).contiguous(), values.expand(-1, -1, 2).contiguous()


def test_default_is_one_sample_per_series():
    x, c = series()
    dataset = DynamicConceptDataset(x, c, ANNOTATIONS)
    assert tuple(dataset.input_data.shape) == (4, 30, 3)
    assert tuple(dataset.concepts.shape) == (4, 30, 2)
    assert torch.equal(dataset.series, torch.arange(4)) and not dataset.start.any()
    assert dataset.n_series == 4


def test_windows_hold_the_steps_of_their_series():
    x, c = series()
    dataset = DynamicConceptDataset(x, c, ANNOTATIONS, window=10, stride=5)
    assert len(dataset) == 5 * 4                                  # 5 windows of each series
    steps = dataset.start[:, None] + torch.arange(10)
    expected = (100 * steps + dataset.series[:, None]).float()
    assert torch.equal(dataset.input_data[..., 0], expected)
    assert torch.equal(dataset.concepts.tensor[..., 0], expected)


def test_sliding_windows_are_a_view_of_the_series():
    x, c = series()
    dataset = DynamicConceptDataset(x, c, ANNOTATIONS, window=10, stride=1)
    assert len(dataset) == 21 * 4
    assert dataset.input_data.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


def test_a_single_series_and_inputs_only():
    x, c = series()
    single = DynamicConceptDataset(x[:, 0], c[:, 0], ANNOTATIONS, window=10, stride=1)
    assert tuple(single.input_data.shape) == (21, 10, 3)
    assert tuple(single.concepts.shape) == (21, 10, 2)
    inputs_only = DynamicConceptDataset(x, window=10)
    assert inputs_only.concepts is None and inputs_only.n_concepts == 0


def test_datamodule_split_in_time_and_max_samples():
    x, c = series(n_steps=60)
    dataset = DynamicConceptDataset(x, c, ANNOTATIONS, window=10, stride=1)
    train = torch.where(dataset.start + 10 <= 40)[0].tolist()
    test = torch.where(dataset.start >= 45)[0].tolist()
    datamodule = ConceptDataModule(dataset, batch_size=8, splitter=FixedIndicesSplitter(train, [], test))
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert tuple(batch["inputs"]["x"].shape) == (8, 10, 3)
    assert batch["concepts"]["c"].annotations.labels == ["c0", "c1"]
    assert (batch["inputs"]["x"] < 100 * 40).all()               # training windows end before step 40
    small = DynamicConceptDataset(x, c, ANNOTATIONS, window=10, stride=1)
    ConceptDataModule(small, max_samples=7, seed=0)
    assert len(small.start) == len(small.series) == 7
    assert torch.equal(small.input_data[:, 0, 0], (100 * small.start + small.series).float())


def test_errors():
    x, c = series()
    with pytest.raises(RuntimeError, match="maximum size"):     # a window longer than the series
        DynamicConceptDataset(x, c, ANNOTATIONS, window=31)
    with pytest.raises(RuntimeError, match="samples"):          # concepts of 3 series, inputs of 4
        DynamicConceptDataset(x, c[:, :3], ANNOTATIONS)
