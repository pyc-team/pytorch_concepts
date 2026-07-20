"""
In this example we show how to run arbitrary conditional queries on a trained concept model, 
using different inference engines.

As in the preovious examples, we use the Asia dataset.

    asia ──► tub ──┐
                   ├──► either ──┬──► xray
    smoke ─► lung ─┘             │
      │                          │
      └───► bronc ───────────────┴──► dysp

The conditional query at test time are:
- P(tub | either, lung)
- P(either | xray)
- P(dysp | lung, bronc)
"""

import itertools

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import ParametricCPD, BayesianNetwork, \
    AncestralSamplingInference, LearnablePrior, RejectionSampling, \
    ImportanceSampling, MutilatedNetworkProposal, PyroImportanceSampling, \
    BeliefPropagation


def _compare_query(
    query_name, evidence_names, targets,
    rejection_engine, torch_is_engine, pyro_is_engine, bp_engine,
):
    """Estimate P(query_name=1 | evidence) with every engine and the data.

    Enumerates every {0,1} assignment of ``evidence_names`` (each a single
    binary node), queries the four inference engines plus the empirical
    frequency in ``targets``, and prints one row per estimator.
    """
    combos = list(itertools.product([0., 1.], repeat=len(evidence_names)))
    rows = {"empirical": [], "reject S": [], "torch IS": [], "pyro  IS": [],
            "belief prop": []}

    for combo in combos:
        evidence = {n: torch.tensor([[v]]) for n, v in zip(evidence_names, combo)}
        query = {query_name: torch.ones(1, 1)}

        rows["reject S"].append(
            rejection_engine.query(query, evidence).probabilities.item())
        rows["torch IS"].append(
            torch_is_engine.query(query, evidence).probabilities.item())
        rows["pyro  IS"].append(
            pyro_is_engine.query(query, evidence).probabilities.item())
        rows["belief prop"].append(
            bp_engine.query(query=[query_name], evidence=evidence)
            .probs[query_name].item())

        mask = torch.ones(targets[query_name].shape[0], dtype=torch.bool)
        for n, v in zip(evidence_names, combo):
            mask = mask & (targets[n].squeeze(-1) == v)
        rows["empirical"].append(
            targets[query_name][mask].mean().item() if mask.any() else float("nan"))

    header = "  ".join(
        "(" + ",".join(f"{n}={int(v)}" for n, v in zip(evidence_names, combo)) + ")"
        for combo in combos
    )
    fmt = lambda vals: "  ".join(f"{v:6.3f}" for v in vals)
    print(f"\n=== P({query_name}=1 | {', '.join(evidence_names)}) ===")
    print(f"{'':>14} | {header}")
    for label in ("empirical", "reject S", "torch IS", "pyro  IS", "belief prop"):
        print(f"{label:>14} | {fmt(rows[label])}")


def main():
    seed_everything(42)

    n_epochs = 800
    n_samples = 10000

    dataset = BnLearnDataset(name='asia', seed=42, n_gen=n_samples)
    # Concepts only: dataset.input_data (embeddings) is not used here.
    concepts = dataset.concepts.float()      # (N, 8), all binary
    names = dataset.graph.node_names         # ['asia','tub','smoke','lung',
                                             #  'bronc','either','xray','dysp']

    # One (N, 1) column per node -- the format `query`/`evidence` expect.
    targets = {
        "asia": concepts[:, 0:1],
        "tub": concepts[:, 1:2],
        "smoke": concepts[:, 2:3],
        "lung": concepts[:, 3:4],
        "bronc": concepts[:, 4:5],
        "either": concepts[:, 5:6],
        "xray": concepts[:, 6:7],
        "dysp": concepts[:, 7:8],
    }

    # Variable setup: one binary concept per node of the ASIA DAG.
    asia_var = ConceptVariable("asia", distribution=Bernoulli, size=1)
    tub_var = ConceptVariable("tub", distribution=Bernoulli, size=1)
    smoke_var = ConceptVariable("smoke", distribution=Bernoulli, size=1)
    lung_var = ConceptVariable("lung", distribution=Bernoulli, size=1)
    bronc_var = ConceptVariable("bronc", distribution=Bernoulli, size=1)
    either_var = ConceptVariable("either", distribution=Bernoulli, size=1)
    xray_var = ConceptVariable("xray", distribution=Bernoulli, size=1)
    dysp_var = ConceptVariable("dysp", distribution=Bernoulli, size=1)

    # ParametricCPD setup: `parents` draws the edges of the DAG. Roots have no
    # parents, so their parametrization is a LearnablePrior (a bare parameter,
    # called with no input). Every other CPD maps its parents' values to a
    # logit; parents are concatenated on the last axis, so `in_features` is
    # the number of parents.
    asia_cpd = ParametricCPD(
        asia_var,
        parametrization={'logits': LearnablePrior(1)},
    )
    smoke_cpd = ParametricCPD(
        smoke_var,
        parametrization={'logits': LearnablePrior(1)},
    )
    tub_cpd = ParametricCPD(
        tub_var,
        parametrization={'logits': nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[asia_var],
    )
    lung_cpd = ParametricCPD(
        lung_var,
        parametrization={'logits': nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[smoke_var],
    )
    bronc_cpd = ParametricCPD(
        bronc_var,
        parametrization={'logits': nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[smoke_var],
    )
    either_cpd = ParametricCPD(
        either_var,
        parametrization={'logits': nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[tub_var, lung_var],
    )
    xray_cpd = ParametricCPD(
        xray_var,
        parametrization={'logits': nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[either_var],
    )
    dysp_cpd = ParametricCPD(
        dysp_var,
        parametrization={'logits': nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                                                 nn.Linear(16, 1))},
        parents=[bronc_var, either_var],
    )

    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[asia_var, tub_var, smoke_var, lung_var,
                   bronc_var, either_var, xray_var, dysp_var],
        factors=[asia_cpd, tub_cpd, smoke_cpd, lung_cpd,
                 bronc_cpd, either_cpd, xray_cpd, dysp_cpd],
    )

    # Inference Initialization: teacher-forced ancestral sampling. Passing the
    # ground truth in `query` both supervises the node and (with p_int=1.0)
    # feeds it to its children.
    inference_engine = AncestralSamplingInference(concept_model, p_int=1.0)

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    inference_engine.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Every node -- roots included -- is now in `query` and `evidence` is
        # empty: nothing is clamped, so the loss below covers the full joint
        # p(asia, smoke, tub, lung, bronc, either, xray, dysp) instead of just
        # the six conditionals p(node | asia, smoke). `out.logits[name]` is
        # always the CPD's own output (the root's `LearnablePrior`, here), so
        # `asia`/`smoke` now get a gradient too. With p_int=1.0 the ground
        # truth still propagates to children exactly as before.
        out = inference_engine.query(
            query={
                "asia": targets["asia"],
                "smoke": targets["smoke"],
                "tub": targets["tub"],
                "lung": targets["lung"],
                "bronc": targets["bronc"],
                "either": targets["either"],
                "xray": targets["xray"],
                "dysp": targets["dysp"],
            },
            evidence={},
        )

        loss = (
            loss_fn(out.logits["asia"], targets["asia"])
            + loss_fn(out.logits["smoke"], targets["smoke"])
            + loss_fn(out.logits["tub"], targets["tub"])
            + loss_fn(out.logits["lung"], targets["lung"])
            + loss_fn(out.logits["bronc"], targets["bronc"])
            + loss_fn(out.logits["either"], targets["either"])
            + loss_fn(out.logits["xray"], targets["xray"])
            + loss_fn(out.logits["dysp"], targets["dysp"])
        )

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.3f}")

    # ---- Test-time: arbitrary conditional queries --------------------------
    # Four engines answer the same queries and are checked against the
    # empirical frequency in the dataset (generated from the true Asia
    # network). ``either`` and ``dysp`` have two parents each, so evidence on
    # only one of them (e.g. ``lung`` for ``either``) still leaves the other
    # parent (``tub``) to be marginalised out -- exactly what these engines
    # are for. ``P(dysp | lung, bronc)`` is answerable by plain ancestral
    # sampling too (lung, bronc are both ancestors of dysp), so it also
    # serves as a sanity check that the backward-capable engines agree with
    # a purely forward computation.
    n_test_samples = 20_000
    rejection_engine = RejectionSampling(concept_model, n_samples=n_test_samples)
    torch_is_engine = ImportanceSampling(
        concept_model, MutilatedNetworkProposal(concept_model),
        n_samples=n_test_samples, initial_temperature=0.1,
    )
    pyro_is_engine = PyroImportanceSampling(concept_model, n_samples=n_test_samples)
    bp_engine = BeliefPropagation(concept_model, iters=20, damping=0.2)

    for query_name, evidence_names in [
        ("tub", ("either", "lung")),
        ("either", ("xray",)),
        ("dysp", ("lung", "bronc")),
    ]:
        _compare_query(
            query_name, evidence_names, targets,
            rejection_engine, torch_is_engine, pyro_is_engine, bp_engine,
        )
        
    return


if __name__ == "__main__":
    main()
