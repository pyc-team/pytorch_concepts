"""Show direct calls, precomputation and graph generators inside a model.

Run: PYTHONPATH=. python examples/utilization/2_model/16_graph_calls.py
The local LLM backend returns A->B; no API key is needed.
"""

import torch
from torch import nn

from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataModule
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.graphs import (
    GraphGeneratorLearnable, GraphGeneratorStatic, remove_weakest_cycles,
)


def backend(prompt, **kwargs):
    return "A->B"


class Model(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.graph_generator = generator

    def forward(self, values):
        graph = self.graph_generator(values, concept_names=["a", "b"])
        return values @ graph.data


def main():
    torch.manual_seed(42)
    values = torch.tensor([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
    names = ["a", "b"]
    descriptions = {"a": "First concept", "b": "Second concept"}
    static = GraphGeneratorStatic("demo", source="LLM", llm_backend=backend)

    # 1. Direct static calls: values, optional names and descriptions.
    graph_automatic_names = static(values)
    assert graph_automatic_names.node_names == ["0", "1"]
    graph_positional = static(values, names, descriptions)
    graph_keyword = static(
        concept_values=values, concept_names=names, concept_descriptions=descriptions,
    )
    torch.testing.assert_close(graph_positional.data, graph_keyword.data)
    # LLM discovery also works with names alone, without observations.
    graph_names_only = static(concept_names=names)
    torch.testing.assert_close(graph_positional.data, graph_names_only.data)
    print("Direct static graph:", graph_positional.data)

    # 2. Dataset precomputation: explicitly supply the training rows.
    dataset = ConceptDataset(
        values, values, annotations=Annotations(labels=names, cardinalities=[1, 1]),
    )
    dataset.precompute_graph(static, training_indices=[0, 1], cache=False)
    torch.testing.assert_close(dataset.graph.data, graph_positional.data)
    print("Dataset graph:", dataset.graph.data)

    # 3. Datamodule precomputation: setup supplies the training rows.
    dm = ConceptDataModule(dataset, val_size=0, test_size=0, seed=42)
    dm.setup("fit")
    dm.precompute_graph(static, cache=False)
    torch.testing.assert_close(dm.dataset.graph.data, graph_positional.data)
    print("Datamodule graph:", dm.dataset.graph.data)

    # 4. Direct learnable calls: DAGMA can also run without observations.
    learned = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=2, refinement=remove_weakest_cycles,
    )
    graph_automatic_names = learned()
    graph_names_only = learned(concept_names=names)
    graph_positional = learned(values, names, descriptions)
    graph_keyword = learned(
        concept_values=values, concept_names=names, concept_descriptions=descriptions,
    )
    assert graph_automatic_names.node_names == ["0", "1"]
    assert graph_names_only.node_names == names
    torch.testing.assert_close(graph_automatic_names.data, graph_positional.data)
    torch.testing.assert_close(graph_positional.data, graph_keyword.data)
    assert graph_positional.data.requires_grad
    print("Direct learnable graph:", graph_positional.data)

    # 5. Inside a model: static graphs are computed in both train and eval.
    static_model = Model(static)
    static_model.train()
    train_output = static_model(values)
    static_model.eval()
    torch.testing.assert_close(static_model(values), train_output)
    assert not train_output.requires_grad
    print("Static model output:", train_output)

    # 6. Inside a model: learnable parameters follow the optimizer and mode.
    learned_model = Model(learned)
    optimizer = torch.optim.SGD(learned_model.parameters(), lr=0.1)
    learned_model.train()
    before = learned.fc1.weight.detach().clone()
    optimizer.zero_grad()
    output = learned_model(values)
    output.square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, learned.fc1.weight)

    learned_model.eval()
    output = learned_model(values)
    assert not output.requires_grad and learned.graph.is_dag()
    print("Learnable model output after one training step:", output)
    print("All usage checks passed. Precomputation accepts only static generators.")


if __name__ == "__main__":
    main()
