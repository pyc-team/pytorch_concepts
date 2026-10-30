"""Tests for WearDataset: a fleet of machines with concepts over time."""
import os

import torch

from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.datasets.wear import ALARM_THRESHOLD, WearDataset


def test_shapes_and_annotations(tmp_path):
    dataset = WearDataset(root=str(tmp_path), n_gen=50, n_steps=10, n_sensors=3)
    assert len(dataset) == 50
    assert dataset.n_series == 50                             # one series per machine: n_gen
    assert dataset.n_features == (10, 5)                      # two operating conditions, then three sensors
    assert dataset.input_data[..., :2].abs().max() <= 1
    assert tuple(dataset.concepts.shape) == (50, 10, 3)     # a time axis: the whole series of each machine
    types = dict(zip(dataset.annotations.labels, dataset.annotations.types))
    assert types == {"health_A": "continuous", "health_B": "continuous", "alarm": "binary"}
    sample = dataset[0]
    assert sample["inputs"]["x"].shape == (10, 5)
    assert sample["concepts"]["c"].shape == (10, 3)
    # the graph of one step: each component's health drives the alarm
    assert int(dataset.graph.data.sum()) == 2


def test_labels_follow_the_law(tmp_path):
    dataset = WearDataset(root=str(tmp_path), n_gen=200, n_steps=8)
    health, alarm = dataset.concepts[["health_A", "health_B"]].tensor, dataset.concepts[["alarm"]].tensor
    assert (health <= 0).all()
    assert (health[:, 1:] <= health[:, :-1]).all()                      # wear is irreversible
    assert torch.equal(alarm, (health.mean(-1, keepdim=True) < ALARM_THRESHOLD).float())
    assert 0 < alarm.mean() < 1


def test_seed_and_cache(tmp_path):
    dataset = WearDataset(root=str(tmp_path), n_gen=30, n_steps=5, seed=0)
    assert os.path.exists(dataset.processed_paths[0])
    reloaded = WearDataset(root=str(tmp_path), n_gen=30, n_steps=5, seed=0)
    assert torch.equal(reloaded.input_data, dataset.input_data)
    assert torch.equal(reloaded.concepts.tensor, dataset.concepts.tensor)
    other = WearDataset(root=str(tmp_path), n_gen=30, n_steps=5, seed=1)
    assert not torch.equal(other.input_data, dataset.input_data)


def test_windows_and_batches(tmp_path):
    series = WearDataset(root=str(tmp_path), n_gen=40, n_steps=12)
    dataset = WearDataset(root=str(tmp_path), n_gen=40, n_steps=12, window=4)
    assert len(dataset) == 3 * 40                             # 3 windows of each machine
    assert torch.equal(dataset.input_data[40 * 2 + 5], series.input_data[5, 8:12])
    datamodule = ConceptDataModule(dataset, batch_size=8, seed=0)
    datamodule.setup("fit")
    concepts = next(iter(datamodule.train_dataloader()))["concepts"]["c"]
    assert tuple(concepts.shape) == (8, 4, 3)
    assert tuple(concepts[["alarm"]].shape) == (8, 4, 1)
