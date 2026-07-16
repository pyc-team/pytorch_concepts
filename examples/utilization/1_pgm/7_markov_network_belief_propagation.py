"""Undirected concept model (a conditional random field) trained with belief
propagation and cross-entropy on the BP marginals.

Two binary concepts ``a`` and ``b`` are coupled by a learnable pairwise
potential, and each has a unary potential conditioned on an observed embedding
``x``. Inference is loopy belief propagation (:class:`BeliefPropagation`), which
returns per-variable marginals; training minimises the cross-entropy of those
marginals against the labels — the partition function ``Z`` is never computed.

This is the undirected counterpart of the directed ``BayesianNetwork`` examples
in this folder, using the same ``query(query, evidence)`` API.
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts import seed_everything, ConceptVariable, EmbeddingVariable
from torch_concepts.nn import (
    MarkovNetwork,
    TabularPotential,
    BeliefPropagation,
    LearnablePrior,
)


def main():
    seed_everything(42)

    n_samples, n_features, n_epochs = 512, 6, 300

    # Data: a = (x0 > 0), b = (x1 > 0). The two concepts are correlated through x.
    x = torch.randn(n_samples, n_features)
    a_lab = (x[:, 0] > 0).long()
    b_lab = (x[:, 1] > 0).long()

    # Variables. The embedding is an observed conditioning input (not in any scope).
    emb = EmbeddingVariable("x", distribution=Normal, size=n_features)
    a = ConceptVariable("a", distribution=Bernoulli, size=1)
    b = ConceptVariable("b", distribution=Bernoulli, size=1)

    # Factors: two conditional unary potentials + one learnable pairwise potential.
    u_a = TabularPotential(scope=[a], parametrization=nn.Linear(n_features, 2),
                           conditioning=[emb], name="u_a")
    u_b = TabularPotential(scope=[b], parametrization=nn.Linear(n_features, 2),
                           conditioning=[emb], name="u_b")
    phi_ab = TabularPotential(scope=[a, b], parametrization=LearnablePrior(2 * 2), name="phi_ab")

    mrf = MarkovNetwork(variables=[emb, a, b], factors=[u_a, u_b, phi_ab])

    # Inference: loopy belief propagation (a few message-passing rounds).
    engine = BeliefPropagation(mrf, iters=5)

    optimizer = torch.optim.AdamW(mrf.parameters(), lr=0.05)
    mrf.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        out = engine.query(query=["a", "b"], evidence={"x": x})
        # BP returns categorical marginals over each variable's states (binary -> 2).
        loss = (
            nn.functional.cross_entropy(out.params["a"]["logits"], a_lab)
            + nn.functional.cross_entropy(out.params["b"]["logits"], b_lab)
        )

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            a_acc = (out.params["a"]["probs"].argmax(-1) == a_lab).float().mean().item()
            b_acc = (out.params["b"]["probs"].argmax(-1) == b_lab).float().mean().item()
            print(f"Epoch {epoch}: Loss {loss.item():.3f} | a acc {a_acc:.2f} | b acc {b_acc:.2f}")


if __name__ == "__main__":
    main()
