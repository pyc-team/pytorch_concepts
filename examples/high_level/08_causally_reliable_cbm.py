"""
Causally Reliable Concept Bottleneck Model
==========================================

A Causally Reliable CBM (C2BM) arranges the concepts along a causal graph:
root concepts are predicted from the input, and every other concept from its
parents in the graph, mixed with an embedding of the input through a
hypernetwork. An intervention on a concept therefore reaches the concepts it
causes, and only those.

Data: the ASIA network (see ``mid_level/05_learning_a_bayesian_network.py``),
whose graph comes with the dataset; the inputs are embeddings of the sampled
concepts.

Reference: De Felice et al., "Causally Reliable Concept Bottleneck Models",
NeurIPS 2025.
"""

# %%
import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    CausallyReliableConceptBottleneckModel,
    ConceptLoss,
    ConceptMetrics,
    DoIntervention,
    UniformPolicy,
    intervention,
)

seed_everything(42)

# %%
# Data
# ----
datamodule = BnLearnDataModule(
    name="asia",
    seed=42,
    batch_size=512,
    root=str(DATA_ROOT / "asia"),
)
graph = datamodule.graph
for node in graph.topological_sort():
    print(f"{node:<7} <- {graph.get_predecessors(node)}")

# %%
# Model and training
# ------------------
model = CausallyReliableConceptBottleneckModel(
    input_size=datamodule.n_features[-1],
    annotations=datamodule.annotations,
    graph=graph,
    backbone=MLP(datamodule.n_features[-1], 64),
    latent_size=64,
    lightning=True,
    loss=ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
    metrics=ConceptMetrics(
        annotations=datamodule.annotations,
        binary={"accuracy": BinaryAccuracy},
        per_concept=True,
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 0.01},
)
trainer = Trainer(max_epochs=50, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

# %%
# Interventions follow the graph
# ------------------------------
# Forcing ``smoke`` changes the predictions of its descendants only.
names = datamodule.annotations.labels
x_test = datamodule.dataset.input_data[datamodule.testset.indices]
model.eval()
with torch.no_grad():
    out = model(query=names, input=x_test)
    with intervention(model, DoIntervention(10.0), UniformPolicy(), ["smoke"]):
        out_do = model(query=names, input=x_test)
changed = [
    n
    for n in names
    if n != "smoke" and not torch.allclose(out.logits[n], out_do.logits[n])
]
print(f"changed by do(smoke): {changed}")
print(f"descendants of smoke: {sorted(graph.get_descendants('smoke'))}")
