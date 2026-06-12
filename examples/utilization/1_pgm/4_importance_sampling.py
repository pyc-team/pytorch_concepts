"""Backward queries with importance sampling on the XOR dataset.

A Bayesian network (C1, C2 -> XOR) is trained forward with ancestral sampling,
then queried backward: evidence on the task leaf (XOR=y), query on the root
concepts (C1, C2). Forward inference cannot answer P(C1,C2|XOR=y) because
there is no forward edge from a leaf to its parents. Importance sampling
proposes the root variables from their priors and reweights by the evidence
likelihood P(XOR=y | C1, C2).

XOR ground truth:
    XOR=1  =>  (C1=0,C2=1) or (C1=1,C2=0)  probability 0.5 each
    XOR=0  =>  (C1=0,C2=0) or (C1=1,C2=1)  probability 0.5 each
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import ToyDataset
from torch_concepts.nn import (
    ParametricCPD, BayesianNetwork,
    AncestralInference, ImportanceSampling, MutilatedNetworkProposal,
    PyroImportanceSampling,
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

    c1_cpd  = ParametricCPD(c1_var)
    c2_cpd  = ParametricCPD(c2_var)
    xor_cpd = ParametricCPD(
        xor_var,
        # XOR is not linearly separable, so a small MLP is needed
        parametrization=nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)),
        parents=[c1_var, c2_var],
    )

    model = BayesianNetwork(
        variables=[c1_var, c2_var, xor_var],
        factors=[c1_cpd, c2_cpd, xor_cpd],
    )

    # ---- Training: forward with ancestral sampling (teacher-forced) ----------
    engine    = AncestralInference(model, p_int=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    bce       = nn.BCELoss()

    model.train()
    print("Training with ancestral sampling ...")
    for epoch in range(500):
        optimizer.zero_grad()
        out = engine.query(
            query={"c1": c_train[:, 0:1], "c2": c_train[:, 1:2], "xor": y_train_oh},
            evidence={},
        )
        c1_pred  = out.params["c1"]["probs"].expand_as(c_train[:, 0:1])
        c2_pred  = out.params["c2"]["probs"].expand_as(c_train[:, 1:2])
        xor_pred = out.params["xor"]["probs"]                   # (N, 2)
        loss = (
            bce(c1_pred,  c_train[:, 0:1])
            + bce(c2_pred,  c_train[:, 1:2])
            + bce(xor_pred, y_train_oh)
        )
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  epoch {epoch:>3}: loss {loss.item():.4f}")
    model.eval()

    # ---- Backward query: P(C1, C2 | XOR=y) ----------------------------------
    # Both engines use the mutilated-network / likelihood-weighting proposal:
    # sample roots from the prior, reweight by P(XOR=y | C1, C2).
    torch_is = ImportanceSampling(
        model, MutilatedNetworkProposal(model),
        n_samples=40_000, initial_temperature=0.1,
    )
    pyro_is = PyroImportanceSampling(model, n_samples=40_000)

    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    c1_q = torch.tensor([[float(a)] for a, _ in combos])       # (4, 1)
    c2_q = torch.tensor([[float(b)] for _, b in combos])       # (4, 1)

    for xor_val in (1, 0):
        # XOR encoding: xor=1 -> one-hot [1,0],  xor=0 -> one-hot [0,1]
        y_ev = torch.zeros(4, 2)
        y_ev[:, 0 if xor_val == 1 else 1] = 1.0

        torch_p = torch_is.query({"c1": c1_q, "c2": c2_q}, {"xor": y_ev}).probabilities
        pyro_p  = pyro_is.query( {"c1": c1_q, "c2": c2_q}, {"xor": y_ev}).probabilities

        # Exact posterior: enumerate all 4 (C1,C2) assignments under trained model
        with torch.no_grad():
            p_c1 = model.name_to_factor("c1")(parent_values={})["probs"].item()
            p_c2 = model.name_to_factor("c2")(parent_values={})["probs"].item()
            joint = []
            for a, b in combos:
                pa   = p_c1 if a == 1 else 1 - p_c1
                pb   = p_c2 if b == 1 else 1 - p_c2
                prbs = model.name_to_factor("xor")(parent_values={
                    "c1": torch.tensor([[float(a)]]),
                    "c2": torch.tensor([[float(b)]]),
                })["probs"]                                     # (1, 2)
                p_y  = prbs[0, 0 if xor_val == 1 else 1].item()
                joint.append(pa * pb * p_y)
            exact = torch.tensor(joint)
            exact = exact / exact.sum()

        header = "  ".join(f"({a},{b})" for a, b in combos)
        fmt    = lambda t: "  ".join(f"{v:6.3f}" for v in t.tolist())
        print(f"\n=== P(C1, C2 | XOR={xor_val}) — evidence on leaf, query on roots ===")
        print(f"{'(C1,C2)':>10} | {header}")
        print(f"{'exact':>10} | {fmt(exact)}")
        print(f"{'torch IS':>10} | {fmt(torch_p)}")
        print(f"{'pyro  IS':>10} | {fmt(pyro_p)}")
        print(f"{'':>10}   torch err {float((torch_p - exact).abs().max()):.3f}"
              f"   pyro err {float((pyro_p - exact).abs().max()):.3f}")


if __name__ == "__main__":
    main()
