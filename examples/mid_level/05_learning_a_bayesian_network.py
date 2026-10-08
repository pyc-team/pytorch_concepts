"""
Learning a Bayesian Network
===========================

Bayesian-network datasets come with their causal graph. Given the graph, each
CPD ``p(node | parents)`` can be learned from data by maximum likelihood, by
feeding every CPD the true values of its parents: this is what
``IndependentInference`` does.

The ASIA network relates a visit to Asia and smoking to tuberculosis, lung
cancer, bronchitis, an X-ray result and dyspnoea::

    asia --> tub ---+
                    +--> either --+--> xray
    smoke -> lung --+             |
      |                           |
      +----> bronc ---------------+--> dysp

Every node is binary: roots get a learnable prior, the other nodes a small MLP
over their parents. The model is built from ``dataset.graph``, so the same code
learns any network of ``BnLearnDataset``.
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, seed_everything
from torch_concepts.data import BnLearnDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    AncestralSamplingInference,
    BayesianNetwork,
    IndependentInference,
    LearnablePrior,
    ParametricCPD,
)

seed_everything(42)

# %%
# Data
# ----
# Only the concepts are used: one (N, 1) column per node.
dataset = BnLearnDataset("asia", n_gen=10000, root=str(DATA_ROOT / "asia"))
graph = dataset.graph
nodes = graph.topological_sort()
train = {n: dataset.concepts[[n]].tensor.float()[:8000] for n in nodes}
test = {n: dataset.concepts[[n]].tensor.float()[8000:] for n in nodes}

# %%
# Model
# -----
variables = {n: ConceptVariable(n, distribution=Bernoulli) for n in nodes}
factors = []
for n in nodes:
    parents = graph.get_predecessors(n)
    if parents:
        layer = torch.nn.Sequential(
            torch.nn.Linear(len(parents), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
    else:
        layer = LearnablePrior(1)
    cpd = ParametricCPD(
        variables[n],
        parents=[variables[p] for p in parents],
        parametrization={"logits": layer},
    )
    factors.append(cpd)
model = BayesianNetwork(variables=list(variables.values()), factors=factors)

# %%
# Training
# --------
# Passing the true values in the query (a dict) lets the engine feed them to
# the children; the loss is the likelihood of every node given its parents.
engine = IndependentInference(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(1000):
    optimizer.zero_grad()
    out = engine.query(train, evidence={})
    loss = sum(loss_fn(out.logits[n], train[n]) for n in nodes)
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
# Sample the learned network (hard samples, ``exact=True``) and compare with
# held-out data: each node's marginal, and dysp given smoke.
sampler = AncestralSamplingInference(model, exact=True)
with torch.no_grad():
    samples = sampler.query(nodes, evidence={}, n_samples=100_000).samples
print(f"{'node':<7} {'P(node) data':>13} {'model':>7}")
for n in nodes:
    print(f"{n:<7} {test[n].mean():>13.3f} {samples[n].mean():>7.3f}")

for value in (1.0, 0.0):
    smoke = torch.full((100_000, 1), value)
    with torch.no_grad():
        dysp = sampler.query(["dysp"], evidence={"smoke": smoke}).samples["dysp"]
    p_data = test["dysp"][test["smoke"] == value].mean()
    print(f"P(dysp | smoke={value:.0f}): data {p_data:.3f}, model {dysp.mean():.3f}")
