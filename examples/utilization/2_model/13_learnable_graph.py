"""Learn a concept bottleneck and a weighted DAGMA-CGM graph on Asia.

Use native dataset concepts and dysp as the task. The illustrative objective
uses concept/task supervision rather than the full DAGMA loss.
"""

# %%
from functools import partial
from pathlib import Path
from unittest.mock import patch

import matplotlib.style as mpl_style
import torch
from torch import nn

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataModule
from torch_concepts.graphs import (
    GraphGeneratorLearnable, initialize_from_entropy, remove_weakest_cycles,
)

seed_everything(42)

# %%
# Data
# ----
# Load CUDA-created caches on CPU without changing BnLearnDataset.
with patch("torch.load", partial(torch.load, map_location="cpu")):
    dm = BnLearnDataModule(
        seed=42, name="asia", root="data/asia_causally_reliable",
        n_gen=2000, batch_size=32,
    )
dm.setup("fit")
print(f"Dataset size: {dm.n_samples} samples")
print(f"Input shape: {dm.n_features}")
print(f"Concepts: {dm.concept_names}")
print(f"Cardinalities: {dm.annotations.cardinalities}")

batch = next(iter(dm.train_dataloader()))
print(f"inputs:   {tuple(batch['inputs']['x'].shape)}")
print(f"concepts: {tuple(batch['concepts']['c'].shape)}")

task_names = ["dysp"]
concept_names = [name for name in dm.concept_names if name not in task_names]
node_indices = [dm.concept_names.index(name) for name in concept_names + task_names]
n_concepts = len(concept_names)

# %%
# Graph
# -----
# Entropy initialization uses only native observations from the training split.
training_values = dm.dataset.concepts.tensor[dm.trainset.indices][:, node_indices]
graph_generator = GraphGeneratorLearnable(
    "dagma_cgm", n_concepts=len(node_indices), n_tasks=len(task_names),
    initialization=initialize_from_entropy(training_values, add_noise=True),
    refinement=remove_weakest_cycles,
)

# %%
# Model
# -----
class GraphConceptBottleneck(nn.Module):
    def __init__(self, input_size, annotations, task_names, graph_generator):
        super().__init__()
        if not isinstance(graph_generator, GraphGeneratorLearnable):
            raise TypeError("graph_generator must be a learnable generator.")
        self.task_names = list(task_names)
        labels = list(annotations.labels)
        if not self.task_names or not set(self.task_names) <= set(labels):
            raise ValueError("task_names must be a non-empty subset of annotation labels.")
        self.concept_names = [name for name in labels if name not in self.task_names]
        self.graph_names = self.concept_names + self.task_names
        n_concepts = len(self.concept_names)
        self.concept_predictor = nn.Linear(input_size, n_concepts)
        self.graph_generator = graph_generator
        self.task_predictor = nn.Linear(n_concepts, len(self.task_names))

    def forward(self, x):
        predicted_concepts = self.concept_predictor(x)
        initial_task = self.task_predictor(predicted_concepts)
        node_values = torch.cat((predicted_concepts, initial_task), dim=1)
        graph = self.graph_generator(node_values, self.graph_names)
        adjacency = graph.data
        n_concepts = len(self.concept_names)
        # Concept edges affect both concept supervision and task predictions.
        concepts = predicted_concepts + predicted_concepts @ adjacency[:n_concepts, :n_concepts]
        tasks = self.task_predictor(concepts) + concepts @ adjacency[:n_concepts, n_concepts:]
        return concepts, tasks, graph


model = GraphConceptBottleneck(
    input_size=dm.n_features[-1], annotations=dm.annotations,
    task_names=task_names, graph_generator=graph_generator,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

# %%
# Training
# --------
model.train()
for epoch in range(20):
    total_loss = 0.0
    for batch in dm.train_dataloader():
        inputs = batch["inputs"]["x"]
        labels = batch["concepts"]["c"].tensor[:, node_indices].float()
        optimizer.zero_grad()
        predicted_concepts, predicted_task, graph = model(inputs)
        loss = (loss_fn(predicted_concepts, labels[:, :n_concepts])
                + loss_fn(predicted_task, labels[:, n_concepts:]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
    print(f"Epoch {epoch + 1}: loss={total_loss / len(dm.trainset):.4f}")

# %%
# Evaluation
# ----------
# Eval removes cycles and caches the final DAG for held-out batches.
dm.setup("test")
model.eval()
concept_correct = task_correct = n_samples = 0
with torch.no_grad():
    for batch in dm.test_dataloader():
        inputs = batch["inputs"]["x"]
        labels = batch["concepts"]["c"].tensor[:, node_indices]
        predicted_concepts, predicted_task, final_graph = model(inputs)
        concept_correct += ((predicted_concepts > 0) == labels[:, :n_concepts].bool()).sum().item()
        task_correct += ((predicted_task > 0) == labels[:, n_concepts:].bool()).sum().item()
        n_samples += len(inputs)
print(f"Test concept accuracy: {concept_correct / (n_samples * n_concepts):.4f}")
print(f"Test task accuracy: {task_correct / (n_samples * len(task_names)):.4f}")
print(f"Final graph: {final_graph.n_nodes} nodes, DAG={final_graph.is_dag()}")
plot_path = Path(__file__).resolve().parents[3] / "outputs" / "learnable_graph" / "final_graph.png"
final_graph.plot(plot_path, rankdir="LR", title="DAGMA - Final Graph")
print(f"Graph image saved to: {plot_path}")
