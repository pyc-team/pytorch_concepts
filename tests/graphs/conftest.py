"""Small datasets shared by graph tests."""

import pytest
import torch
from torch_concepts import Annotations
from torch_concepts import ConceptGraph
from torch_concepts.data.base import ConceptDataModule
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.data.splitters.fixed import FixedIndicesSplitter
from types import SimpleNamespace


@pytest.fixture
def datamodule():
    values = torch.tensor([[0., 1.], [1., 0.], [99., 99.], [88., 88.]])
    dataset = ConceptDataset(
        values.clone(), values.clone(),
        annotations=Annotations(labels=["a", "b"], cardinalities=[1, 1]),
    )
    dataset.label_descriptions = {"a": "description a", "b": "description b"}
    dm = ConceptDataModule(dataset, splitter=FixedIndicesSplitter(
        train_idxs=[0, 1], val_idxs=[2], test_idxs=[3],
    ))
    dm.setup('fit')
    return dm


@pytest.fixture
def dataset():
    names = ["rain", "wet", "traffic"]
    return SimpleNamespace(
        name="toy",
        concept_names=names,
        concepts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        graph_native=ConceptGraph(
            torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            node_names=names,
        ),
        label_descriptions={"rain": "whether it rains", "wet": "wet grass"},
        n_samples=3,
        seed=7,
    )
