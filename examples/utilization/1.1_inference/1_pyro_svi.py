"""
Example: SVI-based Training and Marginal Inference using Pyro

This example demonstrates how to use SVIInference to train a concept-based
model via Stochastic Variational Inference (Pyro's SVI) and then perform
approximate marginal queries.  We build a simple CBM on a job-offer dataset
and compare SVI-derived marginals with empirical frequencies.

Scenario: A student's job offer depends on their studies, which in turn
depend on the economy and their talent.
    [Economy]    [Talent]
        |             |
        └──────┬──────┘
               ▼
           [Studies]
               |
               ▼
           [JobOffer]

Key Features:
- Train CBM parameters through Pyro SVI with TraceEnum_ELBO
- Discrete latent variable (studies) is enumerated exactly
- No guide needed — all latent variables are discrete
- Demonstrate marginal and conditional queries
- Compare SVI estimates with empirical frequencies
"""

import torch
from torch.distributions import Bernoulli

from torch_concepts import LatentVariable, ConceptVariable
from torch_concepts.data.datasets import ToyDAGDataset
from torch_concepts.nn import (
    LinearLatentToConcept,
    LinearConceptToConcept,
    ParametricCPD,
    ProbabilisticModel,
    SVIInference,
    LazyConstructor,
)

NODE_NAMES = ["economy", "talent", "studies", "job_offer"]


def main():
    # ========================================================================
    #  1. Create the synthetic dataset
    # ========================================================================

    latent_dims = 16
    n_svi_steps = 1000
    n_samples = 10000

    dataset = ToyDAGDataset(
        variables=["economy", "talent", "studies", "job_offer"],
        cardinalities={
            "economy": 2, "talent": 2, "studies": 2, "job_offer": 2,
        },
        dag=[
            ("economy", "studies"),
            ("talent", "studies"),
            ("studies", "job_offer"),
        ],
        root_priors={
            "economy": 0.5,
            "talent": 0.5,
        },
        conditional_probs={
            "studies": {
                "economy=0,talent=0": [0.95, 0.05],
                "economy=0,talent=1": [0.95, 0.05],
                "economy=1,talent=0": [0.95, 0.05],
                "economy=1,talent=1": [0.05, 0.95],
            },
            "job_offer": {
                "studies=0": [0.95, 0.05],
                "studies=1": [0.05, 0.95],
            },
        },
        seed=42,
        n_gen=n_samples,
        target_variable="job_offer",
        autoencoder_kwargs={"latent_dim": latent_dims, "epochs": 1000},
        root="data/job_offer_toy_dataset",
    )

    x_train = dataset.input_data
    c_train = dataset.concepts  # (N, 4): economy, talent, studies, job_offer

    print(f"Dataset: {n_samples} samples")
    print(f"Input features: {x_train.shape[1]}")
    print(f"Concepts: {NODE_NAMES}")
    print(f"\nMarginal frequencies:")
    for i, name in enumerate(NODE_NAMES):
        print(f"  P({name}=1) = {c_train[:, i].mean():.3f}")
    print()

    # ========================================================================
    # 2. Define PGM structure
    # ========================================================================

    # REMARK: In this example we use a global learnable latent representation
    # (like a lookup table) instead of per-sample input features.  This
    # simplifies the computation of empirical frequencies for CPT recovery.

    input_var = LatentVariable("input", parents=[], size=latent_dims)
    economy = ConceptVariable("economy", parents=["input"], distribution=Bernoulli)
    talent = ConceptVariable("talent", parents=["input"], distribution=Bernoulli)
    studies = ConceptVariable(
        "studies", parents=["economy", "talent"], distribution=Bernoulli,
    )
    job_offer = ConceptVariable(
        "job_offer", parents=["studies"], distribution=Bernoulli,
    )

    # Learnable global latent input
    class GlobalParams(torch.nn.Module):
        def __init__(self, latent_dims):
            super().__init__()
            self.latent_params = torch.nn.Parameter(torch.randn(1, latent_dims))

        def forward(self, x):
            return self.latent_params.expand(x.shape[0], -1)

    backbone = ParametricCPD("input", parametrization=GlobalParams(latent_dims))
    economy_predictor = ParametricCPD(
        "economy", parametrization=LazyConstructor(LinearLatentToConcept),
    )
    talent_predictor = ParametricCPD(
        "talent", parametrization=LazyConstructor(LinearLatentToConcept),
    )
    studies_predictor = ParametricCPD(
        "studies", parametrization=LazyConstructor(LinearConceptToConcept),
    )
    job_offer_predictor = ParametricCPD(
        "job_offer", parametrization=LazyConstructor(LinearConceptToConcept),
    )

    concept_model = ProbabilisticModel(
        variables=[input_var, economy, talent, studies, job_offer],
        parametric_cpds=[
            backbone,
            economy_predictor,
            talent_predictor,
            studies_predictor,
            job_offer_predictor,
        ],
    )

    # ========================================================================
    # 3. Train the model using SVI
    # ========================================================================

    print("=" * 60)
    print("Training with Pyro SVI")
    print("=" * 60)

    svi_engine = SVIInference(
        concept_model, num_samples=2000, lr=0.005, enumerate_discrete=True,
    )
    initial_input = {"input": x_train}

    # Observation dictionary: economy, talent, and job_offer are observed.
    # Studies is the discrete latent variable — it will be enumerated
    # exactly by TraceEnum_ELBO during training.
    obs_dict = {
        "economy": c_train[:, 0],
        "talent": c_train[:, 1],
        # "studies" is NOT observed (discrete latent)
        "job_offer": c_train[:, 3],
    }

    # build_svi returns (None, svi) when enumerate_discrete=True:
    # no guide is needed because all latent variables are discrete
    # and are summed out exactly via TraceEnum_ELBO.
    _, svi = svi_engine.build_svi()

    # --- SVI training loop ---
    losses = []
    for step in range(n_svi_steps):
        loss = svi.step(initial_input, obs_dict=obs_dict)
        losses.append(loss)
        if step % 500 == 0:
            print(f"  Step {step:4d} — Loss: {loss / n_samples:.4f}")

    print(f"\nFinal ELBO loss: {losses[-1] / n_samples:.4f}")

    # ========================================================================
    # 4. Marginal inference using the trained SVI guide
    # ========================================================================

    print("\n" + "=" * 60)
    print("SVI-based Marginal Inference")
    print("=" * 60)

    concept_model.eval()

    TEST_IDX = 17
    x_test = x_train[TEST_IDX : TEST_IDX + 1]

    target_idx_map = {"economy": 0, "talent": 1, "studies": 2, "job_offer": 3}

    queries = [
        (
            "Q1: p(economy | x)",
            "economy",
            {},
            lambda df: torch.ones(df.shape[0], dtype=torch.bool),
        ),
        (
            "Q2: p(talent | x)",
            "talent",
            {},
            lambda df: torch.ones(df.shape[0], dtype=torch.bool),
        ),
        (
            "Q3: p(job_offer | economy=1, talent=1, x)",
            "job_offer",
            {"economy": torch.ones(1, 1), "talent": torch.ones(1, 1)},
            lambda df: (df[:, 0] == 1) & (df[:, 1] == 1),
        ),
        (
            "Q4: p(talent | job_offer=1, x)",
            "talent",
            {"job_offer": torch.ones(1, 1)},
            lambda df: df[:, 3] == 1,
        )
    ]

    # ── Run queries for increasing sample counts ─────────────────────────
    sample_counts = [100, 500, 1000, 5000]

    # Compute empirical values once
    empirical = {}
    for name, target, obs, mask_fn in queries:
        mask = mask_fn(c_train)
        empirical[name] = (
            c_train[mask, target_idx_map[target]].mean().item()
            if mask.sum() > 0
            else float("nan")
        )

    # results[query_name][n_samples] = p_svi_val
    results = {name: {} for name, *_ in queries}
    for n_s in sample_counts:
        print(f"\nRunning {len(queries)} queries with {n_s} posterior samples...")
        for name, target, obs, _ in queries:
            evidence = {"input": x_test, **obs}
            results[name][n_s] = (
                svi_engine.query(
                    [target], evidence=evidence, num_samples=n_s,
                )
                .squeeze()
                .item()
            )

    # ── Display comparison table ──────────────────────────────────────────
    col_w = 10
    header = (
        f"  {'Query':<42} {'Empirical':>{col_w}}"
        + "".join(f" {'N=' + str(n):>{col_w}}" for n in sample_counts)
    )
    print("\n" + "=" * len(header))
    print(f"{'SVI INFERENCE COMPARISON':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, *_ in queries:
        emp = empirical[name]
        vals = "".join(f" {results[name][n]:>{col_w}.4f}" for n in sample_counts)
        print(f"  {name:<42} {emp:>{col_w}.4f}{vals}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
