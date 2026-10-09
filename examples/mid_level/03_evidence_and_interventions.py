"""
Evidence and Interventions
==========================

Knowledge about a concept can enter a probabilistic model in two ways:

- as *evidence*: the variable, or a single member of a plate, is clamped to the
  given value by name, for every sample;
- as an *intervention*: ``intervention`` overrides the layer of the variable's
  CPD inside a ``with`` block, and a policy can decide per sample which
  concepts to override (e.g. only the least confident).

With a forward engine such as ``DeterministicInference`` both act like the
do-operator: the clamped value flows to the children, not to the parents
(``04_causal_effects.py`` shows the difference between conditioning and
intervening).

The model is the CBM of ``01_concept_bottleneck_model.py`` with a non-linear
task CPD, trained on a noisy reading of the input, so that it makes concept
mistakes for the evidence to correct.
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.distributions import Delta
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    BayesianNetwork,
    DeterministicInference,
    GroundTruthIntervention,
    LearnablePrior,
    LinearEmbeddingToConcept,
    MLPConceptToConcept,
    ParametricCPD,
    UncertaintyInterventionPolicy,
    intervention,
)

seed_everything(42)

# %%
# Data and model
# --------------
dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
x = dataset.input_data + 0.1 * torch.randn(1000, 2)
c = dataset.concepts[["C1", "C2"]]
y = dataset.concepts[["xor"]]
x_train, x_test = x[:800], x[800:]
c_train, c_test = c[:800], c[800:]
y_train, y_test = y[:800], y[800:]

input_var = EmbeddingVariable("input", distribution=Delta, size=2)
latent = EmbeddingVariable("latent", distribution=Delta, size=16)
concepts = ConceptVariable("concepts", members=["C1", "C2"], distribution=Bernoulli)
xor = ConceptVariable("xor", distribution=Bernoulli)

encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
concept_encoder = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=2)
task_predictor = MLPConceptToConcept(in_concepts=2, out_concepts=1, hidden_size=16)

model = BayesianNetwork(
    variables=[input_var, latent, concepts, xor],
    factors=[
        ParametricCPD(input_var, parents=[], parametrization=LearnablePrior(2)),
        ParametricCPD(latent, parents=[input_var], parametrization=encoder),
        ParametricCPD(
            concepts,
            parents=[latent],
            parametrization={"logits": concept_encoder},
        ),
        ParametricCPD(
            xor,
            parents=[concepts],
            parametrization={"logits": task_predictor},
        ),
    ],
)
engine = DeterministicInference(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(500):
    optimizer.zero_grad()
    out = engine.query(["concepts", "xor"], evidence={"input": x_train})
    concept_loss = loss_fn(out.logits["concepts"], c_train)
    task_loss = loss_fn(out.logits["xor"], y_train)
    loss = concept_loss + 0.5 * task_loss
    loss.backward()
    optimizer.step()


def task_accuracy(out):
    return ((out.logits["xor"] > 0).float() == y_test).float().mean().item()


with torch.no_grad():
    no_knowledge = engine.query(["xor"], evidence={"input": x_test})
print(f"no knowledge:               task accuracy {task_accuracy(no_knowledge):.3f}")

# %%
# Evidence
# --------
# Observe one member of the plate, or the whole plate, by name. Evidence takes
# plain tensors: ``.tensor`` unwraps the annotated labels.
on_c1 = {"input": x_test, "C1": c_test[["C1"]].tensor}
on_both = {"input": x_test, "concepts": c_test.tensor}
with torch.no_grad():
    observe_c1 = engine.query(["xor"], evidence=on_c1)
    observe_both = engine.query(["xor"], evidence=on_both)
print(f"evidence on C1:             task accuracy {task_accuracy(observe_c1):.3f}")
print(f"evidence on both concepts:  task accuracy {task_accuracy(observe_both):.3f}")

# %%
# Interventions
# -------------
# Override only the least confident concept of each sample: for the same budget
# of one concept per sample, this beats observing C1 everywhere. The CPD
# outputs logits, so the true values are given as logits.
c_true = torch.logit(c_test, eps=1e-6)
with torch.no_grad(), intervention(
    model,
    GroundTruthIntervention(c_true),
    UncertaintyInterventionPolicy(),
    ["C1", "C2"],
    quantile=0.5,
):
    least_sure = engine.query(["xor"], evidence={"input": x_test})
print(f"least confident concept:    task accuracy {task_accuracy(least_sure):.3f}")
