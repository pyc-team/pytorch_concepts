"""
Plotting the PGM
================

Every high-level model runs on a probabilistic graphical model, stored in its
``pgm`` attribute. ``plot_pgm`` draws it to a file, from top to bottom: one node
per variable (concepts in yellow, embeddings in blue) and an arrow from each parent
to its child. The arrows into a child are parametrized by one layer: they share
its color, and its class name is written once, beside the child (a ``Sequential``
with the modules it chains). The models need no training to be drawn.

The concepts are ``c1`` and ``c2`` and the task ``y``; the C2BM also takes a
causal graph among them. Requires matplotlib.
"""

import torch

from torch_concepts import Annotations, ConceptGraph, seed_everything
from torch_concepts.nn import (
    CausallyReliableConceptBottleneckModel,
    ConceptBottleneckModel,
    ConceptEmbeddingModel,
    ConceptMemoryReasoner,
    plot_pgm,
)

seed_everything(42)

annotations = Annotations(labels=["c1", "c2", "y"], cardinalities=[1, 1, 1])
graph = ConceptGraph(
    torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),  # c1 -> c2, c1 -> y, c2 -> y
    node_names=["c1", "c2", "y"],
)

models = {
    "cbm": ConceptBottleneckModel(input_size=4, annotations=annotations, task_names=["y"]),
    "cem": ConceptEmbeddingModel(input_size=4, annotations=annotations, task_names=["y"]),
    "cmr": ConceptMemoryReasoner(input_size=4, annotations=annotations, task_names=["y"]),
    "c2bm": CausallyReliableConceptBottleneckModel(input_size=4, annotations=annotations, graph=graph),
}

# %%
# A CBM is a chain, ``input -> latent -> concepts -> tasks``: the concepts and the
# tasks are each one plate, a single node. A CEM adds the concept embeddings,
# which feed the tasks directly. A CMR selects a rule from memory for each task:
# the rule roles are learned, with no parent. A C2BM has one node per concept,
# with its own embedding, wired by the causal graph.
for name, model in models.items():
    plot_pgm(model, f"{name}_pgm.pdf")
    print(f"saved {name}_pgm.pdf")
