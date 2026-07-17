"""Backward queries with rejection sampling on the XOR dataset.

A Bayesian network (C1, C2 -> XOR) is trained forward with ancestral sampling,
then queried backward: evidence on the task leaf (XOR=y), query on the root
concepts (C1, C2). Forward inference cannot answer P(C1,C2|XOR=y) because
there is no forward edge from a leaf to its parents. Rejection sampling draws
joint samples from the prior and keeps only those that match the evidence,
estimating P(Q=q | E=e) = |{samples matching Q and E}| / |{samples matching E}|.
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import ToyDataset
from torch_concepts.nn import (
    ParametricCPD, BayesianNetwork, LearnablePrior,
    AncestralSamplingInference, RejectionSampling, Sequential, LearnablePrior
)


def main():
    seed_everything(42)

    # ---- XOR dataset ---------------------------------------------------------
    n_samples = 1000
    dataset     = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx    = list(dataset.graph.edge_index[1].unique().numpy())
    c_train     = dataset.concepts[:, concept_idx]              # (N, 2)  binary
    y_train     = dataset.concepts[:, task_idx]                 # (N, 1)  0/1
    # OneHotCategorical: column 0 = P(XOR=1), column 1 = P(XOR=0)
    y_train_oh  = torch.cat([y_train, 1 - y_train], dim=1)     # (N, 2)

    # ---- Model: C1, C2 (root Bernoulli priors) -> XOR (leaf predictor) -------
    c1_var  = ConceptVariable("c1",  distribution=Bernoulli,         size=1)
    c2_var  = ConceptVariable("c2",  distribution=Bernoulli,         size=1)
    xor_var = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

    # Discrete variables are parametrized by *logits* (an unconstrained real,
    # already in-domain): roots use a LearnablePrior, the leaf an MLP (XOR is
    # not linearly separable). No activation is applied after the parametrization.
    c1_cpd  = ParametricCPD(c1_var, parametrization={"logits": LearnablePrior(1)})
    c2_cpd  = ParametricCPD(c2_var, parametrization={"logits": LearnablePrior(1)})
    xor_cpd = ParametricCPD(
        xor_var,
        parametrization={"logits": nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))},
        parents=[c1_var, c2_var],
    )

    model = BayesianNetwork(
        variables=[c1_var, c2_var, xor_var],
        factors=[c1_cpd, c2_cpd, xor_cpd],
    )

    # ---- Training: forward with ancestral sampling (teacher-forced) ----------
    engine    = AncestralSamplingInference(model, p_int=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    bce       = nn.BCEWithLogitsLoss()

    model.train()
    print("Training with ancestral sampling ...")
    for epoch in range(500):
        optimizer.zero_grad()
        out = engine.query(
            query={"c1": c_train[:, 0:1], "c2": c_train[:, 1:2], "xor": y_train_oh},
            evidence={},
        )
        c1_logit  = out.params["c1"]["logits"].expand_as(c_train[:, 0:1])
        c2_logit  = out.params["c2"]["logits"].expand_as(c_train[:, 1:2])
        xor_logit = out.params["xor"]["logits"]                 # (N, 2)
        loss = (
            bce(c1_logit,  c_train[:, 0:1])
            + bce(c2_logit,  c_train[:, 1:2])
            + bce(xor_logit, y_train_oh)
        )
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  epoch {epoch:>3}: loss {loss.item():.4f}")
    model.eval()

    # ---- Backward query: P(C1, C2 | XOR=y) ----------------------------------
    # Rejection sampling: draw 40k joint samples from the prior, keep those
    # where XOR matches the evidence, count how many also satisfy the query.
    rs = RejectionSampling(model, n_samples=40_000)

    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    c1_q = torch.tensor([[float(a)] for a, _ in combos])       # (4, 1)
    c2_q = torch.tensor([[float(b)] for _, b in combos])       # (4, 1)

    for xor_val in (1, 0):
        # XOR encoding: xor=1 -> one-hot [1,0],  xor=0 -> one-hot [0,1]
        y_ev = torch.zeros(4, 2)
        y_ev[:, 0 if xor_val == 1 else 1] = 1.0

        rs_p = rs.query({"c1": c1_q, "c2": c2_q}, {"xor": y_ev}).probabilities

        # Empirical posterior from the dataset: among the rows whose XOR equals
        # xor_val, the fraction taking each (C1, C2) assignment.
        mask = (y_train[:, 0] == xor_val)
        sub  = c_train[mask]
        emp  = torch.tensor([
            float(((sub[:, 0] == a) & (sub[:, 1] == b)).float().mean())
            for a, b in combos
        ])

        header = "  ".join(f"({a},{b})" for a, b in combos)
        fmt    = lambda t: "  ".join(f"{v:6.3f}" for v in t.tolist())
        print(f"\n=== P(C1, C2 | XOR={xor_val}) — evidence on leaf, query on roots ===")
        print(f"{'(C1,C2)':>10} | {header}")
        print(f"{'empirical':>10} | {fmt(emp)}")
        print(f"{'reject S':>10} | {fmt(rs_p)}")
        print(f"{'':>10}   reject err {float((rs_p - emp).abs().max()):.3f}")

    # ---- Backward query with additional concept evidence: P(C2 | C1=1, XOR=y) -
    # C1=1 is a root variable, so it is clamped during generation (no rejection
    # needed for it). XOR is still a non-root, filtered by rejection.
    c2_targets = torch.tensor([[0.0], [1.0]])              # query C2=0 and C2=1
    c1_ev      = torch.ones(2, 1)                          # C1=1 for both rows

    for xor_val in (1, 0):
        y_ev2 = torch.zeros(2, 2)
        y_ev2[:, 0 if xor_val == 1 else 1] = 1.0

        rs_p2 = rs.query({"c2": c2_targets}, {"c1": c1_ev, "xor": y_ev2}).probabilities

        # Empirical: among rows with C1=1 and XOR=xor_val, the fraction with each C2.
        mask2 = (c_train[:, 0] == 1) & (y_train[:, 0] == xor_val)
        sub2  = c_train[mask2]
        emp2  = torch.tensor([
            float((sub2[:, 1] == c2v).float().mean()) for c2v in (0.0, 1.0)
        ])

        fmt2 = lambda t: "  ".join(f"{v:6.3f}" for v in t.tolist())
        print(f"\n=== P(C2 | C1=1, XOR={xor_val}) — evidence on concept+leaf, query on remaining root ===")
        print(f"{'C2':>10} |  C2=0   C2=1")
        print(f"{'empirical':>10} | {fmt2(emp2)}")
        print(f"{'reject S':>10} | {fmt2(rs_p2)}")
        print(f"{'':>10}   reject err {float((rs_p2 - emp2).abs().max()):.3f}")


if __name__ == "__main__":
    main()
