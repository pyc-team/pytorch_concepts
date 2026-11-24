"""
Symbolic Mixture of Concept Bottleneck Experts
==============================================

A Mixture of Concept Bottleneck Experts (M-CBE) predicts the task with one of
``M`` experts, chosen per sample by a selector::

    input -> latent -> concepts ─┐
                  └──> selector ─┴─> y

``selector`` is a categorical variable over the experts. The task CPD runs
every expert and averages their outputs with the selector value: when the
engine draws the selector as a hard one-hot sample, the average selects a
single expert.

Symbolic M-CBE turns each expert into a formula, in three stages:

1. train the model with MLP experts;
2. fit a symbolic expression to each expert, on the samples routed to it;
3. replace the MLPs with the expressions and fine-tune their constants;
4. (optional) fine tune the constants of the symbolic expressions.

Here ``y = c1 * c2`` or ``y = c1 + c2``, depending on a mechanism hidden in the
input, so two experts recover the two formulas.

Requires ``pip install torch-symbolic`` (PySR installs Julia on first use).

Reference: De Santis et al., "Mixture of Concept Bottleneck Experts", ICML 2026.
"""

# %%
import torch
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough
from symtorch import SymbolicModel

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    MLP,
    AncestralSamplingInference,
    BayesianNetwork,
    LearnablePrior,
    MAPForwardInference,
    MixtureOfExpertsConceptToConcept,
    ParametricCPD,
    SymbolicConceptToConcept,
)

seed_everything(42)

# %%
# Data
# ----
# The input mixes the concepts with the mechanism that picks the formula.
# Mechanisms: c1 + c2, c1 * c2.
n = 2000
c = torch.rand(n, 2) * 2 - 1
mechanism = torch.randint(0, 2, (n, 1)).float()
y = torch.where(mechanism.bool(), c[:, :1] * c[:, 1:], c[:, :1] + c[:, 1:])
x = torch.cat([c, mechanism], dim=-1) @ torch.randn(3, 8)
x_train, x_test = x[:1600], x[1600:]
c_train, c_test = c[:1600], c[1600:]
y_train, y_test = y[:1600], y[1600:]

# %%
# Model
# -----
# Continuous concepts and task are point predictions (``Delta``). The selector
# family draws a hard one-hot sample with a soft gradient (straight-through
# Gumbel-Softmax).
input_var = EmbeddingVariable("input", distribution=Delta, size=8)
latent = EmbeddingVariable("latent", distribution=Delta, size=16)
concepts = ConceptVariable("concepts", members=["c1", "c2"], distribution=Delta)
selector = EmbeddingVariable(
    "selector", distribution=RelaxedOneHotCategoricalStraightThrough, size=2
)
task = ConceptVariable("y", distribution=Delta)

experts = MixtureOfExpertsConceptToConcept([MLP(2, 16, 1), MLP(2, 16, 1)])
model = BayesianNetwork(
    variables=[input_var, latent, concepts, selector, task],
    factors=[
        ParametricCPD(input_var, parents=[], parametrization=LearnablePrior(8)),
        ParametricCPD(latent, parents=[input_var], parametrization=MLP(8, 16)),
        ParametricCPD(concepts, parents=[latent], parametrization={"value": torch.nn.Linear(16, 2)}),
        ParametricCPD(selector, parents=[latent], parametrization={"logits": torch.nn.Linear(16, 2)}),
        ParametricCPD(task, parents=[concepts, selector], parametrization={"value": experts}),
    ],
)

train_engine = AncestralSamplingInference(
    model, initial_temperature=2.0, final_temperature=0.05,
    annealing="exponential", annealing_rate=3e-3,
)
eval_engine = MAPForwardInference(model)  # the most probable expert


def train(optimizer, epochs=1500):
    for _ in range(epochs):
        optimizer.zero_grad()
        out = train_engine.query(["concepts", "y"], evidence={"input": x_train})
        loss = torch.nn.functional.mse_loss(out.value["concepts"], c_train) \
            + torch.nn.functional.mse_loss(out.value["y"], y_train)
        loss.backward()
        optimizer.step()
        train_engine.temperature_step()


def test_mse():
    with torch.no_grad():
        out = eval_engine.query(["y"], evidence={"input": x_test})
    return torch.nn.functional.mse_loss(out.value["y"], y_test).item()


# %%
# Stage 1: train with MLP experts
# -------------------------------
lr = 0.01
train(torch.optim.AdamW(model.parameters(), lr=lr))
print(f"MLP experts      | test MSE {test_mse():.4f}")

# %%
# Stage 2: symbolic regression of each expert
# -------------------------------------------
# Each sample goes to its most probable expert; SymTorch distils the expert on
# its own samples. NOTE: ground-truth concepts as inputs, as in the paper.
with torch.no_grad():
    route = eval_engine.query(["selector"], evidence={"input": x_train}).logits["selector"].argmax(-1)
# NOTE: serial PySR; multi-threaded search can deadlock while Julia compiles.
# "score" keeps the simplest formula, not one that also fits the MLP's noise.
sr_params = {"binary_operators": ["+", "-", "*"], "unary_operators": [],
             "niterations": 40, "parallelism": "serial", "model_selection": "score"}
for m, expert in enumerate(experts.experts):
    regressor = SymbolicModel(expert, block_name=f"expert_{m}").distill(
        c_train[route == m], sr_params=sr_params,
        fit_params={"variable_names": ["c1", "c2"]},
    )[0]
    experts.experts[m] = SymbolicConceptToConcept([regressor.sympy()], ["c1", "c2"])

# %%
# Stage 3: fine-tune the expressions
# ----------------------------------
# Only the constants of the formulas are trainable now, so the learning rate
# is 5x higher, as in the paper.
train(torch.optim.AdamW(model.parameters(), lr=5 * lr), epochs=500)
print(f"symbolic experts | test MSE {test_mse():.4f}")
for m, expert in enumerate(experts.experts):
    print(f"expert {m}: y = {expert.sympy()[0]}")
