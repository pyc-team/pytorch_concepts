"""
Markov Network
==============

In an undirected model, concepts are coupled by potentials instead of being
linked by causal arrows, so evidence on one concept informs the others in any
direction. Here two binary concepts ``a`` and ``b`` depend on an observed input
``x`` and on a shared hidden cause: even given ``x``, knowing ``a`` changes the
odds of ``b``. A model that predicts each concept from the input alone cannot
use this; a single potential over ``a``, ``b`` and ``x`` can.

``BeliefPropagation`` computes marginals by message passing. The model is
trained on the likelihood of the joint ``p(a, b | x) = p(a | x) p(b | a, x)``,
where both factors are queries, the second with ``a`` as evidence. (Fitting
``p(a | x)`` and ``p(b | x)`` separately would not identify the coupling.)
"""

# %%
import torch
from torch.distributions import Bernoulli, Normal

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.nn import BeliefPropagation, MarkovNetwork, ParametricPotential

seed_everything(42)

# %%
# Data
# ----
x = torch.randn(4000, 2)
hidden = torch.randn(4000, 1)  # not observed; shared by both concepts
a = (x[:, :1] + hidden > 0).float()
b = (x[:, 1:] + hidden > 0).float()
x_train, x_test = x[:3000], x[3000:]
a_train, a_test = a[:3000], a[3000:]
b_train, b_test = b[:3000], b[3000:]

# %%
# Model
# -----
# One potential over all three variables: an energy network scores every
# configuration of ``a`` and ``b`` given ``x``. The observed input is a
# variable of the network like any other, held fixed by evidence.
x_var = EmbeddingVariable("x", distribution=Normal, size=2)
a_var = ConceptVariable("a", distribution=Bernoulli)
b_var = ConceptVariable("b", distribution=Bernoulli)
energy = torch.nn.Sequential(
    torch.nn.Linear(1 + 1 + 2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)
model = MarkovNetwork(
    variables=[x_var, a_var, b_var],
    factors=[ParametricPotential(scope=[a_var, b_var, x_var], parametrization=energy)],
)
engine = BeliefPropagation(model, iters=5)

# %%
# Training
# --------
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
bce = torch.nn.functional.binary_cross_entropy
for epoch in range(500):
    optimizer.zero_grad()
    p_a = engine.query(["a"], evidence={"x": x_train}).probs["a"]
    p_b_given_a = engine.query(["b"], evidence={"x": x_train, "a": a_train}).probs["b"]
    loss = bce(p_a, a_train) + bce(p_b_given_a, b_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
# Evidence on ``a`` sharpens the prediction of ``b``.
with torch.no_grad():
    p_b = engine.query(["b"], evidence={"x": x_test}).probs["b"]
    p_b_given_a = engine.query(["b"], evidence={"x": x_test, "a": a_test}).probs["b"]
accuracy = ((p_b > 0.5).float() == b_test).float().mean()
accuracy_given_a = ((p_b_given_a > 0.5).float() == b_test).float().mean()
print(f"accuracy on b, given x:       {accuracy:.3f}")
print(f"accuracy on b, given x and a: {accuracy_given_a:.3f}")

# Where x says little (both inputs near 0), b follows a, in the model as in the data.
near = (x_test.abs() < 0.5).all(dim=-1, keepdim=True)
for value in (1.0, 0.0):
    rows = near & (a_test == value)
    model_p, data_p = p_b_given_a[rows].mean(), b_test[rows].mean()
    print(f"P(b=1 | a={value:.0f}, x near 0): model {model_p:.2f}, data {data_p:.2f}")
