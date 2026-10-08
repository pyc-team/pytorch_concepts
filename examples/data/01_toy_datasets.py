"""
Toy Datasets
============

Toy datasets are small synthetic problems with known concepts, generated on
first use and cached in the PyC cache (``~/.cache/pyc``). Each one holds input
features, concept labels (the tasks are stored as concepts too) and a graph
over the concepts.

- ``ToyDataset``: ``xor``, ``trigonometry`` and ``dot``, where a task is
  computed from two or three concepts, and ``checkmark``, whose concepts follow
  a small causal graph.
- ``CompletenessDataset``: tasks generated from concepts, some of which can be
  hidden from the model, to study what happens when the concepts are not
  enough to explain the task.
"""

# %%
from torch_concepts.data import CompletenessDataset, ToyDataset
from torch_concepts.env import DATA_ROOT

# %%
# Toy datasets
# ------------
for name in ["xor", "trigonometry", "dot", "checkmark"]:
    dataset = ToyDataset(name, n_gen=1000, root=str(DATA_ROOT / name))
    graph = dataset.graph
    edges = [f"{s}->{t}" for s in graph.node_names for t in graph.get_successors(s)]
    print(name)
    print(f"  inputs:   {tuple(dataset.input_data.shape)}")
    print(f"  concepts: {dataset.concept_names}")
    print(f"  graph:    {edges}")

# %%
# Incomplete concepts
# -------------------
# Four concepts generate the task, but only three of them are given.
dataset = CompletenessDataset(
    name="completeness",
    root=str(DATA_ROOT / "completeness"),
    n_gen=1000,
    n_concepts=3,
    n_hidden_concepts=1,
    n_tasks=1,
)
print("completeness")
print(f"  inputs:   {tuple(dataset.input_data.shape)}")
print(f"  concepts: {dataset.concept_names}")
