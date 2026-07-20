"""Concept model over the ASIA DAG, trained with ancestral sampling.

The classic ASIA (chest-clinic) network from bnlearn::

    asia ──► tub ──┐
                   ├──► either ──┬──► xray
    smoke ─► lung ─┘             │
      │                          │
      └───► bronc ───────────────┴──► dysp

Every node is a binary concept, so this is a "standard CBM" whose concept layer
is *not* flat: instead of predicting all concepts independently from an input,
each concept is predicted from its **parents in the DAG**. The two roots
(``asia``, ``smoke``) have no parents and are parametrized by a
:class:`LearnablePrior`; every other node gets a small MLP over its parents.

No embeddings are used: ``dataset.input_data`` (the autoencoder features the
dataset also generates) is deliberately ignored — the model is purely a
concept-to-concept graph.

Training uses :class:`AncestralSamplingInference` with ``p_int=1.0``: the roots
are given as ``evidence`` and every other CPD sees the *ground-truth* parents
(teacher forcing), which is exactly maximum likelihood for each conditional
``p(node | parents)``.

At test time the same roots are clamped as evidence but ``p_int=0.0`` makes the
engine propagate its **own** samples down the DAG, so the estimate of
``p(node | asia, smoke)`` reflects how information actually flows through the
graph. It is compared against the empirical frequencies in the dataset.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import ParametricCPD, BayesianNetwork, \
    AncestralSamplingInference, LearnablePrior


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

    # The roots are given, exactly as they will be at test time: they go in
    # `evidence`, not in `query`. An observed variable emits no parameters, so
    # there is no loss term for `asia`/`smoke` -- the model only learns the six
    # conditionals p(node | parents).
    root_evidence = {
        "asia": targets["asia"],
        "smoke": targets["smoke"],
    }

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    inference_engine.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # one logit per queried node; out.logits[name] slices a (N, 1) view
        out = inference_engine.query(
            query={
                "tub": targets["tub"],
                "lung": targets["lung"],
                "bronc": targets["bronc"],
                "either": targets["either"],
                "xray": targets["xray"],
                "dysp": targets["dysp"],
            },
            evidence=root_evidence,
        )

        loss = (
            loss_fn(out.logits["tub"], targets["tub"])
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

    # ---- Test-time: p(non-roots | roots) ----------------------------------
    # Same evidence as during training -- the roots observed in the data -- but
    # p_int=0.0 now makes the engine propagate its *own* samples instead of the
    # ground truth, so information really flows down the DAG: every node is
    # sampled from parents that are themselves samples. A low temperature keeps
    # the relaxed samples close to hard 0/1.
    non_roots = ["tub", "lung", "bronc", "either", "xray", "dysp"]
    propagation_engine = AncestralSamplingInference(
        concept_model, p_int=0.0, initial_temperature=0.1
    )
    with torch.no_grad():
        propagated = propagation_engine.query(
            query=non_roots,
            evidence=root_evidence,
        )

    # One generated sample per data row, so `rows` selects the matching subset
    # on both sides. The two roots are binary, so there are four configurations.
    for asia_value, smoke_value in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]:
        rows = ((targets["asia"] == asia_value)
                & (targets["smoke"] == smoke_value)).squeeze(-1)

        print(f"\np(node | asia={asia_value:.0f}, smoke={smoke_value:.0f})"
              f"   [{int(rows.sum())} data rows]")
        for name in non_roots:
            p_data = targets[name][rows].mean().item()
            p_model = (propagated.samples[name][rows] > 0.5).float().mean().item()
            print(f"  {name:8s} data {p_data:.3f} | model {p_model:.3f}")

    return


if __name__ == "__main__":
    main()
