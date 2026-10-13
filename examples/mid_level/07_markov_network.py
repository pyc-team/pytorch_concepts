"""
Markov Network
==============

In an undirected model, variables are coupled by potentials instead of being
linked by directed edges. 
Here three binary neurons: ``N1`` and ``N2`` fire at random, and
``N3`` computes ``N1 AND N2``, with 2.5% noise. A single potential over all
three learns the gate.

The model is trained by **pseudo-likelihood**: each variable is predicted from
all the others, ``p(N1 | N2, N3) * p(N2 | N1, N3) * p(N3 | N1, N2)``. Each factor is
one ``BeliefPropagation`` query, and a value in the query makes the engine
report its probability in ``out.probabilities``. (Fitting each marginal
``p(N3)`` on its own would not identify the gate.)

The trained model is then queried by belief propagation
and by exact variable elimination.
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, seed_everything
from torch_concepts.nn import (
    BeliefPropagation,
    MarkovNetwork,
    ParametricPotential,
    PgmpyVariableElimination,
)

seed_everything(42)

# %%
# Data
# ----
n1 = torch.bernoulli(torch.full((1000, 1), 0.5))
n2 = torch.bernoulli(torch.full((1000, 1), 0.5))
flip = torch.rand(1000, 1) < 0.025
n3 = torch.where(flip, 1 - n1 * n2, n1 * n2)
data = {"N1": n1, "N2": n2, "N3": n3}

# %%
# Model
# -----
# A unary potential on each input neuron, and one potential over all three
# that scores every configuration of the gate.
N1 = ConceptVariable("N1", distribution=Bernoulli)
N2 = ConceptVariable("N2", distribution=Bernoulli)
N3 = ConceptVariable("N3", distribution=Bernoulli)
model = MarkovNetwork(
    variables=[N1, N2, N3],
    factors=[
        ParametricPotential(scope=[N1], parametrization=torch.nn.Linear(1, 1, bias=False)),
        ParametricPotential(scope=[N2], parametrization=torch.nn.Linear(1, 1, bias=False)),
        ParametricPotential(
            scope=[N1, N2, N3],
            parametrization=torch.nn.Sequential(
                torch.nn.Linear(3, 10), torch.nn.LeakyReLU(0.1), torch.nn.Linear(10, 1)
            ),
        ),
    ],
)
engine = BeliefPropagation(model, iters=10)

# %%
# Training
# --------
# The negative pseudo-log-likelihood: one query per variable, the others as
# evidence, the observed value as the query's target.
names = ["N1", "N2", "N3"]
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(500):
    optimizer.zero_grad()
    loss = 0.0
    for name in names:
        others = {other: data[other] for other in names if other != name}
        p = engine.query({name: data[name]}, evidence=others).probabilities
        loss = loss - p.log().mean()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Queries
# -------
# First qurey: ``P(N3=1 | N1, N2)``
# Second query: ``P(N1=1, N2=1 | N3=1)``

# First query using loopy BP
Probs = engine(query={'N3':torch.Tensor([1,1,1,1])}, evidence={"N1":torch.Tensor([0,0,1,1]), "N2":torch.Tensor([0,1,0,1])}).probabilities
Probs = [x.item() for x in list(Probs)]
print("P(N3 | N1, N2)", Probs)

# Second query using BP. This engine cannot estimate conditionals with multiple variables in the query.
# Therefore, we apply the chain rule.
p_N1_given_N3 = engine(query={'N1':torch.Tensor([1])}, evidence={"N3":torch.Tensor([1])}).probabilities.item()
p_N2_given_N3_N1 = engine(query={'N2':torch.Tensor([1])}, evidence={"N3":torch.Tensor([1]),'N1':torch.Tensor([1])}).probabilities.item()
print("P(N1=1, N2=2 | N3=1)", p_N1_given_N3 * p_N2_given_N3_N1)

# Now the variable elimination engine
engine = PgmpyVariableElimination(model)

# First query
Probs = engine(query={'N3':torch.Tensor([1,1,1,1])}, evidence={"N1":torch.Tensor([0,0,1,1]), "N2":torch.Tensor([0,1,0,1])}).probabilities
Probs = [x.item() for x in list(Probs)]
print("P(N3 | N1, N2)", Probs)

# Second query
Probs = engine(query={'N1':torch.Tensor([1]), 'N2':torch.Tensor([1])}, evidence={"N3":torch.Tensor([1])}).probabilities.item()
print("P(N1=1, N2=2 | N3=1)", Probs)


# %%
