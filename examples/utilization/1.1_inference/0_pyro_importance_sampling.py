"""
Example: Sampling-based Marginal Inference using Pyro

This example demonstrates how to use ImportanceSamplingInference to perform
approximate marginal queries on trained concept-based models. We train a simple
CBM on a job-offer dataset and then use Pyro-based importance sampling to
estimate marginal probabilities.

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
- Train CBM using IndependentInference
- Use ImportanceSamplingInference for marginal queries
- Demonstrate forward, conditional, and backward (explaining-away) queries
- Compare importance sampling estimates with empirical frequencies
"""

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli
from torchmetrics import Accuracy

from torch_concepts import LatentVariable, ConceptVariable
from torch_concepts.data.datasets import ToyDAGDataset
from torch_concepts.nn import (
    LinearLatentToConcept,
    LinearConceptToConcept,
    ParametricCPD,
    ProbabilisticModel,
    ImportanceSamplingInference,
    IndependentInference,
    LazyConstructor
)

NODE_NAMES = ["economy", "talent", "studies", "job_offer"]

def main():
    # ========================================================================
    #  1. Create the synthetic dataset
    # ========================================================================

    # DAG: economy -> studies, talent -> studies, studies -> job_offer
    # CPTs match the notebook mixing_pyro_CBMs.ipynb:
    #   P(economy=1) = 0.10,  P(talent=1) = 0.75
    #   P(studies=1 | e, t) ≈ 0.05 + 0.10*e + 0.10*t + 0.70*e*t
    #   P(job_offer=1 | studies=0) = 0.05,  P(job_offer=1 | studies=1) = 0.80

    latent_dims = 16
    n_epochs = 1000
    n_samples = 10000

    dataset = ToyDAGDataset(
        variables=['economy', 'talent', 'studies', 'job_offer'],
        cardinalities={'economy': 2, 'talent': 2, 'studies': 2, 'job_offer': 2},
        dag=[('economy', 'studies'), ('talent', 'studies'), ('studies', 'job_offer')],
        root_priors={
            'economy': 0.5,
            'talent': 0.5,
        },
        conditional_probs={
            'studies': {
                "economy=0,talent=0": [0.95, 0.05],
                "economy=0,talent=1": [0.95, 0.05],
                "economy=1,talent=0": [0.95, 0.05],
                "economy=1,talent=1": [0.05, 0.95],
            },
            'job_offer': {
                "studies=0": [0.95, 0.05],
                "studies=1": [0.05, 0.95],
            }
        },
        seed=42,
        n_gen=n_samples,
        target_variable='job_offer',
        autoencoder_kwargs={'latent_dim': latent_dims, 'epochs': 1000},
        root="data/job_offer_toy_dataset"
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
    # 2. Define PGM structure and train the Concept Model
    # ========================================================================

    # REMARK: In this example, we get rid of the input "x" and just learn a global latent representation for the dataset (like a lookup table).
    # This simplifies the computation of empirical frequencies for the CPT recovery analysis.

    input_var = LatentVariable("input", parents=[], size=latent_dims)
    economy = ConceptVariable("economy", parents=["input"], distribution=Bernoulli)
    talent = ConceptVariable("talent", parents=["input"], distribution=Bernoulli)
    studies = ConceptVariable("studies", parents=["economy", "talent"], distribution=Bernoulli)
    job_offer = ConceptVariable("job_offer", parents=["studies"], distribution=Bernoulli)

    # Create a torch model that in the forward just returns a learnable torch.parameter with dimension (bsz, latent_dims).
    class global_params(torch.nn.Module):
        def __init__(self, latent_dims):
            super().__init__()
            self.latent_params = torch.nn.Parameter(torch.randn(1, latent_dims))

        def forward(self, x):
            return self.latent_params.expand(x.shape[0], -1)

    # Define CPDs (neural networks)
    backbone = ParametricCPD(
        "input",
        parametrization=global_params(latent_dims=latent_dims)
    )

    economy_predictor = ParametricCPD(
        "economy",
        parametrization=LazyConstructor(LinearLatentToConcept)
    )
    talent_predictor = ParametricCPD(
        "talent",
        parametrization=LazyConstructor(LinearLatentToConcept)
    )
    studies_predictor = ParametricCPD(
        "studies",
        parametrization=LazyConstructor(LinearConceptToConcept)
    )
    job_offer_predictor = ParametricCPD(
        "job_offer",
        parametrization=LazyConstructor(LinearConceptToConcept)
    )

    concept_model = ProbabilisticModel(
        variables=[input_var, economy, talent, studies, job_offer],
        parametric_cpds=[backbone, economy_predictor, talent_predictor,
                         studies_predictor, job_offer_predictor]
    )

    # Train using IndependentInference
    inference_engine = IndependentInference(concept_model)
    initial_input = {'input': x_train}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        c_pred = inference_engine.query(
            NODE_NAMES, evidence=initial_input, ground_truth=c_train, concept_names=NODE_NAMES, debug=True
        )
        loss = loss_fn(c_pred, c_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            concept_accuracy = accuracy_score(c_train.numpy(), (c_pred > 0.).detach().numpy())

    # ========================================================================
    # 3. Marginal inference using Importance Sampling
    # ========================================================================

    # Note: Importance sampling (Monte Carlo approximation) is not exact inference!
    # Accuracy improves with more samples: error ~ O(1/√num_samples)

    print("\n" + "=" * 60)
    print("Sampling-based Marginal Inference with Pyro")
    print("=" * 60)

    concept_model.eval()

    # Use a single test point 
    TEST_IDX = 17
    x_test = x_train[TEST_IDX:TEST_IDX + 1]
    true_vars = {name: c_train[TEST_IDX, i].item() for i, name in enumerate(NODE_NAMES)}

    # Only needed if the model was trained with input features and we want to condition on them during inference (not global params). 
    # print(f"\nTest point (index {TEST_IDX}):")
    # for k, v in true_vars.items():
    #     print(f"  {k} = {int(v)}")

    # ── Define the 5 test queries (matching the notebook) ─────────────────

    target_idx_map = {"economy": 0, "talent": 1, "studies": 2, "job_offer": 3}

    queries = [
        ("Q1: p(economy | x)",             "economy",   {},
         lambda df: torch.ones(df.shape[0], dtype=torch.bool)),
        ("Q2: p(talent | x)",              "talent",    {},
         lambda df: torch.ones(df.shape[0], dtype=torch.bool)),
        ("Q3: p(studies | x)",                       "studies",    {},
         lambda df: torch.ones(df.shape[0], dtype=torch.bool)),
        ("Q4: p(studies | economy=1, x)",             "studies",    {"economy": torch.ones(1, 1)},
         lambda df: df[:, 0] == 1),
        ("Q5: p(economy | job_offer=1, x)",           "economy",   {"job_offer": torch.ones(1, 1)},
         lambda df: df[:, 3] == 1),
        ("Q6: p(studies | job_offer=1, x)",           "studies",    {"job_offer": torch.ones(1, 1)},
         lambda df: df[:, 3] == 1),
        ("Q7: p(job_offer | economy=1, talent=1, x)", "job_offer", {"economy": torch.ones(1, 1), "talent": torch.ones(1, 1)},
         lambda df: (df[:, 0] == 1) & (df[:, 1] == 1)),
    ]

    # ── Run queries for increasing sample counts ─────────────────────────
    sample_counts = [100, 500, 1000, 5000]

    # Compute empirical values once
    empirical = {}
    for name, target, obs, mask_fn in queries:
        mask = mask_fn(c_train)
        empirical[name] = c_train[mask, target_idx_map[target]].mean().item() if mask.sum() > 0 else float('nan')

    # results[query_name][n_samples] = p_is_val
    results = {name: {} for name, *_ in queries}
    for n_s in sample_counts:
        print(f"\nRunning {len(queries)} queries with {n_s} importance samples...")
        sampler = ImportanceSamplingInference(concept_model, num_samples=n_s, num_draws=n_s)
        for name, target, obs, _ in queries:
            evidence = {'input': x_test, **obs}
            results[name][n_s] = sampler.query([target], evidence=evidence).squeeze().item()

    # ── Display comparison table ──────────────────────────────────────────
    col_w = 10
    header = f"  {'Query':<42} {'Empirical':>{col_w}}" + "".join(f" {'N='+str(n):>{col_w}}" for n in sample_counts)
    print("\n" + "=" * len(header))
    print(f"{'INFERENCE METHOD COMPARISON':^{len(header)}}")
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
