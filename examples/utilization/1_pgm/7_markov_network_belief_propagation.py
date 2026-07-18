"""Undirected concept model (a conditional random field) trained with belief
propagation and cross-entropy on the BP marginals.

Two binary concepts ``a`` and ``b`` are coupled by a learnable pairwise
potential, and each has a unary potential conditioned on an observed embedding
``x``. Inference is loopy belief propagation (:class:`BeliefPropagation`), which
returns per-variable marginals in each variable's own parametrization; training
minimises the cross-entropy of those marginals against the labels — the
partition function ``Z`` is never computed.

This is the undirected counterpart of the directed ``BayesianNetwork`` examples
in this folder, using the same ``query(query, evidence)`` API.
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts import seed_everything, ConceptVariable, EmbeddingVariable
from torch_concepts.nn import MarkovNetwork, ParametricPotential, BeliefPropagation


def main():
    seed_everything(42)

    n_samples, n_features, n_epochs = 512, 6, 300

    # Data: a = (x0 > 0), b = (x1 > 0). The two concepts are correlated through x.
    x = torch.randn(n_samples, n_features)
    # Targets match each concept's own parametrization: a width-1 float per
    # binary variable, exactly as for a directed BayesianNetwork.
    a_lab = (x[:, 0] > 0).float().unsqueeze(-1)
    b_lab = (x[:, 1] > 0).float().unsqueeze(-1)

    # Variables. The embedding is an observed conditioning input (not in any scope).
    emb = EmbeddingVariable("x", distribution=Normal, size=n_features)
    a = ConceptVariable("a", distribution=Bernoulli, size=1)
    b = ConceptVariable("b", distribution=Bernoulli, size=1)

    # A single energy-based potential over both concepts, conditioned on the
    # embedding: E(a, b ; x) = MLP([onehot(a), onehot(b), x]). The MLP lets x
    # shape each concept AND couples a with b (a linear energy would be additive).
    energy = nn.Sequential(
        nn.Linear(1 + 1 + n_features, 128), nn.ReLU(), nn.Linear(128, 1)
    )
    phi = ParametricPotential(scope=[a, b], parametrization=energy,
                              conditioning=[emb], name="phi")

    mrf = MarkovNetwork(variables=[emb, a, b], factors=[phi])

    # Inference: loopy belief propagation (a few message-passing rounds).
    engine = BeliefPropagation(mrf, iters=5)

    optimizer = torch.optim.AdamW(mrf.parameters(), lr=0.05)
    mrf.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        out = engine.query(query=["a", "b"], evidence={"x": x})
        # BP reports each marginal in the variable's own parametrization, so a
        # binary concept yields Bernoulli logits (log-odds) of width 1 — the
        # same loss you would write against a directed engine's output.
        loss = (
            nn.functional.binary_cross_entropy_with_logits(out.params["a"]["logits"], a_lab)
            + nn.functional.binary_cross_entropy_with_logits(out.params["b"]["logits"], b_lab)
        )

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            a_acc = ((out.params["a"]["probs"] > 0.5).float() == a_lab).float().mean().item()
            b_acc = ((out.params["b"]["probs"] > 0.5).float() == b_lab).float().mean().item()
            print(f"Epoch {epoch}: Loss {loss.item():.3f} | a acc {a_acc:.2f} | b acc {b_acc:.2f}")


if __name__ == "__main__":
    main()
