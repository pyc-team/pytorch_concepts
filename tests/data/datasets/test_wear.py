"""Tests for WearDataset: a fleet of machines with concepts over time."""
import os
import shutil
import tempfile

import pytest
import torch

from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.datasets.wear import ALARM_THRESHOLD, WearDataset


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_shapes_and_annotations(temp_dir):
    dataset = WearDataset(root=temp_dir, n_gen=50, window=6, horizon=4, n_sensors=3)
    assert len(dataset) == 50
    assert dataset.n_features == (6, 5)                       # two operating conditions, then three sensors
    assert dataset.input_data[..., :2].abs().max() <= 1
    assert tuple(dataset.concepts.shape) == (50, 5, 3)      # a time axis: the window's last step and 4 more
    types = dict(zip(dataset.annotations.labels, dataset.annotations.types))
    assert types == {"health_A": "continuous", "health_B": "continuous", "alarm": "binary"}
    sample = dataset[0]
    assert sample["inputs"]["x"].shape == (6, 5)
    assert sample["concepts"]["c"].shape == (5, 3)
    # the graph of one step: each component's health drives the alarm
    assert int(dataset.graph.data.sum()) == 2


def test_labels_follow_the_law(temp_dir):
    dataset = WearDataset(root=temp_dir, n_gen=200, horizon=5)
    health, alarm = dataset.concepts[["health_A", "health_B"]].tensor, dataset.concepts[["alarm"]].tensor
    assert (health <= 0).all()
    assert (health[:, 1:] <= health[:, :-1]).all()                      # wear is irreversible
    assert torch.equal(alarm, (health.mean(-1, keepdim=True) < ALARM_THRESHOLD).float())
    assert 0 < alarm.mean() < 1


def test_seed_and_cache(temp_dir):
    dataset = WearDataset(root=temp_dir, n_gen=30, horizon=2, seed=0)
    assert os.path.exists(dataset.processed_paths[0])
    reloaded = WearDataset(root=temp_dir, n_gen=30, horizon=2, seed=0)
    assert torch.equal(reloaded.input_data, dataset.input_data)
    assert torch.equal(reloaded.concepts.tensor, dataset.concepts.tensor)
    other = WearDataset(root=temp_dir, n_gen=30, horizon=2, seed=1)
    assert not torch.equal(other.input_data, dataset.input_data)


def test_batches_keep_the_time_axis(temp_dir):
    dataset = WearDataset(root=temp_dir, n_gen=40, horizon=3)
    datamodule = ConceptDataModule(dataset, batch_size=8, seed=0)
    datamodule.setup("fit")
    concepts = next(iter(datamodule.train_dataloader()))["concepts"]["c"]
    assert tuple(concepts.shape) == (8, 4, 3)
    assert tuple(concepts[["alarm"]].shape) == (8, 4, 1)
