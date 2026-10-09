"""
Concept Bottleneck Model as a Probabilistic Model
=================================================

The mid-level API describes a model as a probabilistic graphical model: random
variables connected by conditional probability distributions (CPDs), each
parametrized by a neural network. A Concept Bottleneck Model is the chain::

    input -> latent -> concepts -> xor

where ``concepts`` is a *plate*: a single variable holding the binary concepts
``C1`` and ``C2``, produced by one layer and addressable together or one by
one. An inference engine answers queries on the model; ``DeterministicInference``
is a standard forward pass.

As in ``low_level/02_concept_bottleneck_model.py``, the task CPD is linear and
cannot represent XOR; ``02_concept_embedding_model.py`` lifts this limit.

Reference: Koh et al., "Concept Bottleneck Models", ICML 2020.
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
    LearnablePrior,
    LinearConceptToConcept,
    LinearEmbeddingToConcept,
    ParametricCPD,
)

seed_everything(42)

# %%
# Data
# ----
dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
x = dataset.input_data
c = dataset.concepts[["C1", "C2"]]
y = dataset.concepts[["xor"]]
x_train, x_test = x[:800], x[800:]
c_train, c_test = c[:800], c[800:]
y_train, y_test = y[:800], y[800:]

# %%
# Variables
# ---------
# Embeddings are deterministic (``Delta``); concepts are random variables.
input_var = EmbeddingVariable("input", distribution=Delta, size=2)
latent = EmbeddingVariable("latent", distribution=Delta, size=16)
concepts = ConceptVariable("concepts", members=["C1", "C2"], distribution=Bernoulli)
xor = ConceptVariable("xor", distribution=Bernoulli)

# %%
# Model
# -----
# Each CPD maps the values of its ``parents`` to the parameters of its
# variable's distribution (here the logits of a Bernoulli). The root ``input``
# gets a prior, but it is always observed.
encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
concept_encoder = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=2)
task_predictor = LinearConceptToConcept(in_concepts=2, out_concepts=1)

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

# %%
# Training
# --------
# A query asks for variables given evidence; the output holds the parameters of
# each queried variable, by name.
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
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
# Plate members are addressable by name. Observing the concepts as evidence
# gives the task CPD's answer for each combination of them.
with torch.no_grad():
    out = engine.query(["C1", "C2"], evidence={"input": x_test})
    combinations = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    task = engine.query(["xor"], evidence={"concepts": combinations})
    p_xor = torch.sigmoid(task.logits["xor"])
for name in ["C1", "C2"]:
    accuracy = ((out.logits[name] > 0).float() == c_test[[name]]).float().mean()
    print(f"test accuracy of {name}: {accuracy:.2f}")
for (c1, c2), p in zip(combinations.int().tolist(), p_xor.flatten().tolist()):
    print(f"C1={c1}, C2={c2} -> P(xor) = {p:.2f}  (true: {c1 ^ c2})")
