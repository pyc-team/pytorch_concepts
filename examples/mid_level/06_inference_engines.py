"""
Inference Engines
=================

An inference engine answers queries ``P(query | evidence)`` on a probabilistic
model, and the right engine depends on the question.

- Forward engines run the model from causes to effects.
  ``DeterministicInference`` propagates probabilities: fast and differentiable,
  it is the engine to train with, but it only approximates the marginals of
  downstream nodes. ``AncestralSamplingInference`` propagates samples and
  ``MAPForwardInference`` the most likely value of each node.
- Evidence on effects, such as a symptom, has to flow back to its causes.
  ``RejectionSampling`` and ``ImportanceSampling`` estimate such queries by
  sampling, ``BeliefPropagation`` by message passing, and
  ``PgmpyVariableElimination`` computes them exactly.

The model is the ASIA network of ``05_learning_a_bayesian_network.py``. The
query is a diagnosis: given a patient with dyspnoea and a positive X-ray, how
likely are lung cancer and bronchitis?
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
    BeliefPropagation,
    DeterministicInference,
    ImportanceSampling,
    IndependentInference,
    LearnablePrior,
    MAPForwardInference,
    MutilatedNetworkProposal,
    ParametricCPD,
    PgmpyVariableElimination,
    RejectionSampling,
)

seed_everything(42)

# %%
# Data and model
# --------------
# As in ``05_learning_a_bayesian_network.py``.
dataset = BnLearnDataset("asia", n_gen=10000, root=str(DATA_ROOT / "asia"))
graph = dataset.graph
nodes = graph.topological_sort()
data = {n: dataset.concepts[[n]].tensor.float() for n in nodes}

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

trainer = IndependentInference(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(1000):
    optimizer.zero_grad()
    out = trainer.query(data, evidence={})
    loss = sum(loss_fn(out.logits[n], data[n]) for n in nodes)
    loss.backward()
    optimizer.step()

# %%
# Forward engines
# ---------------
# From a cause to an effect: how likely is a positive X-ray for a smoker?
smoker = {"smoke": torch.ones(1, 1)}
smokers = {"smoke": torch.ones(100_000, 1)}
with torch.no_grad():
    out = DeterministicInference(model).query(["xray"], evidence=smoker)
    deterministic = torch.sigmoid(out.logits["xray"]).item()
    ancestral = AncestralSamplingInference(model, exact=True)
    sampled = ancestral.query(["xray"], evidence=smokers).samples["xray"].mean().item()
    out = PgmpyVariableElimination(model).query(["xray"], evidence=smoker)
    exact = out.probs["xray"].item()
    out = MAPForwardInference(model).query(["xray"], evidence=smoker)
    most_likely = out.samples["xray"].item()
print("P(xray | smoker):")
print(f"  deterministic       {deterministic:.3f}")
print(f"  ancestral sampling  {sampled:.3f}")
print(f"  exact (pgmpy)       {exact:.3f}")
print(f"  most likely value   {most_likely:.0f}")

# %%
# A diagnosis
# -----------
# From effects back to their causes, compared with the frequency in the data.
evidence = {"dysp": torch.ones(1, 1), "xray": torch.ones(1, 1)}
rows = (data["dysp"] == 1) & (data["xray"] == 1)
engines = {
    "rejection sampling": RejectionSampling(model, n_samples=100_000),
    # a low temperature makes the relaxed samples, and so the estimate, close to exact
    "importance sampling": ImportanceSampling(
        model,
        MutilatedNetworkProposal(model),
        n_samples=100_000,
        initial_temperature=0.01,
    ),
    "belief propagation": BeliefPropagation(model, iters=20, damping=0.2),
    "exact (pgmpy)": PgmpyVariableElimination(model),
}
causes = ["lung", "bronc"]
print(f"\nP(cause | dyspnoea, positive x-ray), {int(rows.sum())} patients in the data")
print(f"{'':<20}" + "".join(f"{c:>8}" for c in causes))
print(f"{'data':<20}" + "".join(f"{data[c][rows].mean():>8.3f}" for c in causes))
with torch.no_grad():
    for name, engine in engines.items():
        if isinstance(engine, (RejectionSampling, ImportanceSampling)):
            # sampling engines score a query assignment: P(cause = 1 | evidence)
            outs = [engine.query({c: torch.ones(1, 1)}, evidence) for c in causes]
            p = [out.probabilities.item() for out in outs]
        else:
            p = [engine.query([c], evidence=evidence).probs[c].item() for c in causes]
        print(f"{name:<20}" + "".join(f"{v:>8.3f}" for v in p))
