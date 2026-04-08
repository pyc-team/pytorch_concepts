"""
Example: Differentiable Variable Elimination for Exact Inference
================================================================

Demonstrates training a concept-based Bayesian Network using differentiable
Variable Elimination (VE) and subsequently using VE for exact conditional
queries at test time.

Scenario — Job Offer Model
---------------------------
::

    [Economy]  [Talent]
         \\      /
        [Studies]
            |
       [JobOffer]

All variables are binary (Bernoulli).  The model learns the full CPTs
(prior probabilities for root nodes, conditional probability tables for
children) by maximising the log-likelihood of observed data via VE.

Training
--------
1. Build discrete factor tables from the neural-network parametrised CPDs
   (fully differentiable).
2. Use VE with no evidence to compute the joint distribution
   P(economy, talent, studies, job_offer).
3. For every training sample, index into the joint tensor to obtain its
   probability, then minimise the negative log-likelihood.
4. Gradients flow through VE's factor products and marginalisations back
   to the neural network weights.

Test-Time Queries
-----------------
Use VE to compute exact conditional distributions such as:

- P(studies) — marginal probability
- P(studies | economy=1) — forward query
- P(economy | job_offer=1) — explaining away
"""

import torch
import numpy as np
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, VariableEliminationInference


# ── Ground truth CPTs ────────────────────────────────────────────────
GT_P_ECONOMY = 0.7          # P(economy=1)
GT_P_TALENT = 0.6           # P(talent=1)
# P(studies=1 | economy, talent)
GT_P_STUDIES = {(0, 0): 0.1, (0, 1): 0.4, (1, 0): 0.5, (1, 1): 0.9}
# P(job_offer=1 | studies)
GT_P_JOB = {0: 0.2, 1: 0.8}

NODE_NAMES = ["economy", "talent", "studies", "job_offer"]


# ── Data generation ──────────────────────────────────────────────────

def generate_data(n_samples: int, seed: int = 42) -> torch.Tensor:
    """Sample from the ground-truth Bayesian Network.

    Returns a (n_samples, 4) float tensor with columns
    [economy, talent, studies, job_offer] ∈ {0, 1}.
    """
    rng = np.random.RandomState(seed)
    data = np.empty((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        e = int(rng.random() < GT_P_ECONOMY)
        t = int(rng.random() < GT_P_TALENT)
        s = int(rng.random() < GT_P_STUDIES[(e, t)])
        j = int(rng.random() < GT_P_JOB[s])
        data[i] = [e, t, s, j]

    return torch.from_numpy(data)


# ── Model definition ─────────────────────────────────────────────────

def build_model() -> ProbabilisticModel:
    """Construct a ProbabilisticModel for the job-offer DAG.

    Root nodes (economy, talent) use ``Linear(1, 1)``; the output is
    ``W·0 + b = b``, so the bias alone parameterises the prior logit.

    Child nodes use a ``Linear(n_parent_features, 1)`` that maps the
    concatenated binary parent states to a child logit.
    """
    # Variables
    economy = ConceptVariable("economy", distribution=Bernoulli)
    talent = ConceptVariable("talent", distribution=Bernoulli)
    studies = ConceptVariable("studies", distribution=Bernoulli)
    job_offer = ConceptVariable("job_offer", distribution=Bernoulli)

    # CPDs
    cpd_economy = ParametricCPD("economy",
                                parametrization=torch.nn.Linear(1, 1))
    cpd_talent = ParametricCPD("talent",
                               parametrization=torch.nn.Linear(1, 1))
    cpd_studies = ParametricCPD("studies",
                                parametrization=torch.nn.Linear(2, 1),
                                parents=["economy", "talent"])
    cpd_job = ParametricCPD("job_offer",
                            parametrization=torch.nn.Linear(1, 1),
                            parents=["studies"])

    model = ProbabilisticModel(
        variables=[economy, talent, studies, job_offer],
        factors=[cpd_economy, cpd_talent, cpd_studies, cpd_job],
    )
    return model


# ── Training via VE ──────────────────────────────────────────────────

def train(model: ProbabilisticModel, data: torch.Tensor,
          n_epochs: int = 2000, lr: float = 0.05):
    """Train the BN by maximising log-likelihood using Variable Elimination.

    At each step VE computes the full joint distribution
    P(economy, talent, studies, job_offer) as a (2,2,2,2) tensor.
    The NLL of each sample is obtained by indexing into this tensor.
    """
    ve = VariableEliminationInference(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    idx = data.long()  # (N, 4) with values in {0, 1}

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Joint distribution via VE (no evidence → full joint)
        _Z, joint = ve.query(query=NODE_NAMES, evidence={})

        # log P for every sample
        log_joint = torch.log(joint.values.clamp(min=1e-10))
        sample_log_probs = log_joint[idx[:, 0], idx[:, 1],
                                     idx[:, 2], idx[:, 3]]
        loss = -sample_log_probs.mean()

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:4d}  NLL = {loss.item():.4f}")


# ── CPT recovery ─────────────────────────────────────────────────────

def show_learned_cpts(model: ProbabilisticModel):
    """Print the learned CPTs and compare with ground truth."""
    print("\n" + "=" * 60)
    print("Learned CPTs vs Ground Truth")
    print("=" * 60)

    with torch.no_grad():
        # Root priors (Linear(1,1) with zero input → output = bias)
        dummy = torch.zeros(1, 1)

        p_e = torch.sigmoid(
            model.factors["economy"].parametrization(input=dummy)
        ).item()
        print(f"\nP(economy=1):   learned = {p_e:.4f}   "
              f"truth = {GT_P_ECONOMY}")

        p_t = torch.sigmoid(
            model.factors["talent"].parametrization(input=dummy)
        ).item()
        print(f"P(talent=1):    learned = {p_t:.4f}   "
              f"truth = {GT_P_TALENT}")

        # Studies CPT
        parent_states = torch.tensor(
            [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
        )
        p_s = torch.sigmoid(
            model.factors["studies"].parametrization(input=parent_states)
        ).squeeze(-1)
        print("\nP(studies=1 | economy, talent):")
        for i, (e, t) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            print(f"  e={e}, t={t}:  learned = {p_s[i].item():.4f}   "
                  f"truth = {GT_P_STUDIES[(e, t)]}")

        # JobOffer CPT
        parent_states = torch.tensor([[0.], [1.]])
        p_j = torch.sigmoid(
            model.factors["job_offer"].parametrization(input=parent_states)
        ).squeeze(-1)
        print("\nP(job_offer=1 | studies):")
        for s_val in [0, 1]:
            print(f"  s={s_val}:  learned = {p_j[s_val].item():.4f}   "
                  f"truth = {GT_P_JOB[s_val]}")


# ── Empirical queries from data ───────────────────────────────────────

# Column indices matching NODE_NAMES order.
COL = {name: i for i, name in enumerate(NODE_NAMES)}


def _empirical_cond(data: torch.Tensor, query_col: int,
                    query_val: int,
                    evidence: dict) -> float:
    """Compute P(query_col = query_val | evidence) from raw counts."""
    mask = torch.ones(data.size(0), dtype=torch.bool)
    for col, val in evidence.items():
        mask &= data[:, col] == val
    subset = data[mask]
    if subset.size(0) == 0:
        return float('nan')
    return (subset[:, query_col] == query_val).float().mean().item()


def _empirical_joint_cond(data: torch.Tensor,
                          query_cols: list,
                          query_vals: list,
                          evidence: dict) -> float:
    """Compute P(query_cols = query_vals | evidence) from raw counts."""
    mask = torch.ones(data.size(0), dtype=torch.bool)
    for col, val in evidence.items():
        mask &= data[:, col] == val
    subset = data[mask]
    if subset.size(0) == 0:
        return float('nan')
    match = torch.ones(subset.size(0), dtype=torch.bool)
    for c, v in zip(query_cols, query_vals):
        match &= subset[:, c] == v
    return match.float().mean().item()


# ── Test-time queries ────────────────────────────────────────────────

def exact_queries(model: ProbabilisticModel, data: torch.Tensor):
    """Run exact VE queries and compare with empirical estimates."""
    ve = VariableEliminationInference(model)

    print("\n" + "=" * 60)
    print("VE Queries vs Empirical Estimates")
    print("=" * 60)

    with torch.no_grad():

        # 1. Marginals
        print("\n--- Marginal probabilities ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        for var in NODE_NAMES:
            _Z, result = ve.query(query=[var], evidence={})
            ve_p = result.values[1].item()
            emp_p = _empirical_cond(data, COL[var], 1, {})
            print(f"  P({var}=1){'':<35s} {ve_p:8.4f}  {emp_p:9.4f}")

        # 2. Forward queries
        print("\n--- Forward queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")

        queries_fwd = [
            ("studies", {"economy": 1}),
            ("studies", {"economy": 1, "talent": 1}),
            ("job_offer", {"studies": 1}),
        ]
        for qvar, ev in queries_fwd:
            _Z, r = ve.query(query=[qvar], evidence=ev)
            ve_p = r.values[1].item()
            emp_ev = {COL[k]: v for k, v in ev.items()}
            emp_p = _empirical_cond(data, COL[qvar], 1, emp_ev)
            ev_str = ", ".join(f"{k}={v}" for k, v in ev.items())
            label = f"P({qvar}=1 | {ev_str})"
            print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")

        # 3. Explaining away
        print("\n--- Explaining-away queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")

        queries_ea = [
            ("economy", {"job_offer": 1}),
            ("talent", {"job_offer": 1}),
            ("economy", {"job_offer": 1, "talent": 1}),
        ]
        for qvar, ev in queries_ea:
            _Z, r = ve.query(query=[qvar], evidence=ev)
            ve_p = r.values[1].item()
            emp_ev = {COL[k]: v for k, v in ev.items()}
            emp_p = _empirical_cond(data, COL[qvar], 1, emp_ev)
            ev_str = ", ".join(f"{k}={v}" for k, v in ev.items())
            label = f"P({qvar}=1 | {ev_str})"
            print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")

        # 4. Joint conditional
        print("\n--- Joint conditional queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        _Z, r = ve.query(query=["economy", "talent"],
                         evidence={"job_offer": 1})
        emp_ev = {COL["job_offer"]: 1}
        for e in range(2):
            for t in range(2):
                ve_p = r.values[e, t].item()
                emp_p = _empirical_joint_cond(
                    data, [COL["economy"], COL["talent"]], [e, t], emp_ev)
                label = f"P(economy={e}, talent={t} | job_offer=1)"
                print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    n_samples = 5000
    n_epochs = 2000
    lr = 0.05

    print("Generating data from ground-truth BN ...")
    data = generate_data(n_samples, seed=42)

    print(f"\nDataset: {n_samples} samples, {len(NODE_NAMES)} binary nodes")
    print(f"Empirical frequencies:  "
          f"economy={data[:, 0].mean():.3f}  "
          f"talent={data[:, 1].mean():.3f}  "
          f"studies={data[:, 2].mean():.3f}  "
          f"job_offer={data[:, 3].mean():.3f}")

    print("\nBuilding model ...")
    model = build_model()

    print(f"\nTraining via differentiable VE ({n_epochs} epochs) ...")
    model.train()
    train(model, data, n_epochs=n_epochs, lr=lr)

    model.eval()
    show_learned_cpts(model)
    exact_queries(model, data)


if __name__ == "__main__":
    main()
