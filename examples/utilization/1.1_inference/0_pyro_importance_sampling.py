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

def show_learned_cpts(concept_model, x_train, c_train):
    """Probe learned CPDs and compare against empirical frequencies computed from the dataset."""
    concept_model.eval()
    with torch.no_grad():
        # ── Root nodes: average predicted probability over the dataset ────
        economy_cpd = concept_model.get_module_of_concept("economy")
        p_economy_learned = torch.sigmoid(economy_cpd.parametrization(x_train)).mean().item()

        talent_cpd = concept_model.get_module_of_concept("talent")
        p_talent_learned = torch.sigmoid(talent_cpd.parametrization(x_train)).mean().item()

        # ── Studies CPT: probe all 4 (economy, talent) combinations ───────
        # LinearConceptToConcept applies sigmoid internally, so pass large
        # logits to simulate hard 0/1 parent activations.
        L = 10.0
        studies_cpd = concept_model.get_module_of_concept("studies")
        studies_inputs = torch.tensor([[-L, -L], [-L, L], [L, -L], [L, L]])
        p_studies_learned = torch.sigmoid(
            studies_cpd.parametrization(studies_inputs)
        ).squeeze(-1).tolist()

        # ── Job offer CPT: probe studies=0 and studies=1 ──────────────────
        job_cpd = concept_model.get_module_of_concept("job_offer")
        job_inputs = torch.tensor([[-L], [L]])
        p_job_learned = torch.sigmoid(
            job_cpd.parametrization(job_inputs)
        ).squeeze(-1).tolist()

    # Empirical frequencies from the dataset
    p_economy_emp = c_train[:, 0].mean().item()
    p_talent_emp  = c_train[:, 1].mean().item()

    print("\n" + "=" * 70)
    print(f"{'CPT RECOVERY ANALYSIS':^70}")
    print("=" * 70)

    print("\n── Root nodes ──")
    print(f"  {'Quantity':<30} {'Learned':>8}  {'Empirical':>9}  {'Error':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*9}  {'-'*8}")
    for name, learned, emp in [
        ("P(economy=1)", p_economy_learned, p_economy_emp),
        ("P(talent=1)",  p_talent_learned,  p_talent_emp),
    ]:
        err = learned - emp
        print(f"  {name:<30} {learned:>8.4f} {emp:>9.4f}  {err:>+8.4f}")

    print("\n── Studies CPT  P(studies=1 | economy, talent) ──")
    print(f"  {'Condition':<30} {'Learned':>8} {'Empirical':>9}  {'Error':>8}")
    print(f"  {'-'*30} {'-'*8}   {'-'*9}  {'-'*8}")
    study_labels = ["e=0, t=0", "e=0, t=1", "e=1, t=0", "e=1, t=1"]
    study_masks = [
        (c_train[:, 0] == 0) & (c_train[:, 1] == 0),
        (c_train[:, 0] == 0) & (c_train[:, 1] == 1),
        (c_train[:, 0] == 1) & (c_train[:, 1] == 0),
        (c_train[:, 0] == 1) & (c_train[:, 1] == 1),
    ]
    for label, learned, mask in zip(study_labels, p_studies_learned, study_masks):
        emp = c_train[mask, 2].mean().item() if mask.sum() > 0 else float('nan')
        err = learned - emp
        print(f"  {label:<30} {learned:>8.4f} {emp:>9.4f}  {err:>+8.4f}")

    print("\n── Job Offer CPT  P(job_offer=1 | studies) ──")
    print(f"  {'Condition':<30} {'Learned':>8}  {'Empirical':>9}  {'Error':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*9}  {'-'*8}")
    for label, learned, s_val in [
        ("studies=0", p_job_learned[0], 0),
        ("studies=1", p_job_learned[1], 1),
    ]:
        mask = c_train[:, 2] == s_val
        emp = c_train[mask, 3].mean().item() if mask.sum() > 0 else float('nan')
        err = learned - emp
        print(f"  {label:<30} {learned:>8.4f} {emp:>9.4f}  {err:>+8.4f}")
    print("\n" + "=" * 70)


def main():
    # ========================================================================
    #  1. Create the synthetic dataset
    # ========================================================================

    # DAG: economy -> studies, talent -> studies, studies -> job_offer

    latent_dims = 16
    n_epochs = 5000
    n_samples = 100000

    dataset = ToyDAGDataset(
        variables=['economy', 'talent', 'studies', 'job_offer'],
        cardinalities={'economy': 2, 'talent': 2, 'studies': 2, 'job_offer': 2},
        dag=[('economy', 'studies'), ('talent', 'studies'), ('studies', 'job_offer')],
        root_priors={
            'economy': 0.95,
            'talent': 0.05,
        },
        conditional_probs={
            'studies': {
                "economy=0,talent=0": [0.90, 0.10],
                "economy=0,talent=1": [0.20, 0.80],
                "economy=1,talent=0": [0.20, 0.80],
                "economy=1,talent=1": [0.05, 0.95],
            },
            'job_offer': {
                "studies=0": [0.90, 0.10],
                "studies=1": [0.10, 0.90],
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
        parametrization=torch.nn.Sequential(
            torch.nn.Linear(latent_dims, latent_dims),
            torch.nn.LeakyReLU(),
        ) #global_params(latent_dims=latent_dims)
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

    loss_history = []
    accuracy_history = []

    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        c_pred = inference_engine.query(
            NODE_NAMES,
            evidence=initial_input,
            ground_truth=c_train,
            concept_names=NODE_NAMES,
            debug=True,
            return_logits=True,
        )
        loss = loss_fn(c_pred, c_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            concept_accuracy = accuracy_score(c_train.numpy(), (c_pred > 0.).detach().numpy())
            loss_history.append(loss.item())
            accuracy_history.append(concept_accuracy)

    print(f"Training losses:", loss_history)
    print(f"Training accuracies:", accuracy_history)

    # ========================================================================
    # 2b. CPT Recovery Analysis
    # ========================================================================
    # Probe each learned CPD with all parent combinations and compare
    # against the empirical frequencies computed from the dataset. 
    show_learned_cpts(concept_model, x_train, c_train)

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
