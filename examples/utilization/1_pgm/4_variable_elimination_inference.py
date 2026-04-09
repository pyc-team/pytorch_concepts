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

All variables are binary (Bernoulli).  A ``ToyDAGDataset`` generates
samples from the ground-truth BN and produces autoencoder embeddings.
The model takes each sample's embedding as input and predicts the
concept values through input-conditioned CPDs, trained by maximising
the log-likelihood via differentiable VE.

Training
--------
1. Each CPD's neural network takes the input embedding (concatenated
   with parent-state features for child nodes) and outputs logits.
2. VE multiplies the per-sample factors to compute the per-sample
   joint distribution P(economy, talent, studies, job_offer | x).
3. NLL loss is the negative log of the joint entry corresponding to
   the observed concept values.
4. Gradients flow through VE back to the CPD network weights.

Test-Time Queries
-----------------
Use VE (without input conditioning) to compute exact conditional
distributions such as:

- P(studies) — marginal probability
- P(studies | economy=1) — forward query
- P(economy | job_offer=1) — explaining away
"""

import torch
import numpy as np
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, VariableEliminationInference
from torch_concepts.data.datasets.categorical_toy_dag import ToyDAGDataset

# ── Ground truth CPTs ────────────────────────────────────────────────
GT_P_ECONOMY = 0.7
GT_P_TALENT = 0.6
GT_P_STUDIES = {(0, 0): 0.1, (0, 1): 0.4, (1, 0): 0.5, (1, 1): 0.9}
GT_P_JOB = {0: 0.2, 1: 0.8}

NODE_NAMES = ["economy", "talent", "studies", "job_offer"]
COL = {name: i for i, name in enumerate(NODE_NAMES)}

N_SAMPLES = 5000
N_EPOCHS = 2000
LR = 0.05
EMB_DIM = 8


def main():
    # ── 1. Generate dataset via ToyDAGDataset ────────────────────────
    print("Generating data via ToyDAGDataset ...")

    cpt_studies = np.zeros((2, 2, 2))
    for (e, t), p in GT_P_STUDIES.items():
        cpt_studies[1, e, t] = p
        cpt_studies[0, e, t] = 1.0 - p

    cpt_job = np.zeros((2, 2))
    for s, p in GT_P_JOB.items():
        cpt_job[1, s] = p
        cpt_job[0, s] = 1.0 - p

    dataset = ToyDAGDataset(
        variables=NODE_NAMES,
        cardinalities={n: 2 for n in NODE_NAMES},
        dag=[("economy", "studies"), ("talent", "studies"),
             ("studies", "job_offer")],
        conditional_probs={("studies",): cpt_studies,
                           ("studies", "job_offer"): cpt_job},
        root_priors={"economy": np.array([1 - GT_P_ECONOMY, GT_P_ECONOMY]),
                     "talent": np.array([1 - GT_P_TALENT, GT_P_TALENT])},
        seed=42,
        n_gen=N_SAMPLES,
        autoencoder_kwargs={"latent_dim": EMB_DIM, "epochs": 200},
    )

    # Extract embeddings and concept labels
    embeddings = dataset.input_data          # (N, EMB_DIM)
    concepts = dataset.concepts              # (N, n_concepts)
    data = concepts.float()                  # concept labels as float

    print(f"\nDataset: {N_SAMPLES} samples, embedding dim = {EMB_DIM}")
    print(f"Empirical frequencies:  "
          f"economy={data[:, 0].mean():.3f}  "
          f"talent={data[:, 1].mean():.3f}  "
          f"studies={data[:, 2].mean():.3f}  "
          f"job_offer={data[:, 3].mean():.3f}")

    # ── 2. Build model (input-conditioned CPDs) ──────────────────────
    print("\nBuilding model ...")
    model = ProbabilisticModel(
        variables=[ConceptVariable("economy", distribution=Bernoulli),
                   ConceptVariable("talent", distribution=Bernoulli),
                   ConceptVariable("studies", distribution=Bernoulli),
                   ConceptVariable("job_offer", distribution=Bernoulli)],
        factors=[
            ParametricCPD("economy",
                          parametrization=torch.nn.Linear(EMB_DIM, 1)),
            ParametricCPD("talent",
                          parametrization=torch.nn.Linear(EMB_DIM, 1)),
            ParametricCPD("studies",
                          parametrization=torch.nn.Linear(2, 1),
                          parents=["economy", "talent"]),
            ParametricCPD("job_offer",
                          parametrization=torch.nn.Linear(1, 1),
                          parents=["studies"]),
        ],
    )

    # ── 3. Train via VE with input embeddings ────────────────────────
    print(f"\nTraining via differentiable VE ({N_EPOCHS} epochs) ...")
    model.train()
    ve = VariableEliminationInference(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    idx = data.long()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        out = ve.query(query=NODE_NAMES, evidence={'input': embeddings},
                       return_log_joint=True)
        log_joint = out['log_joint']  # (N, 2, 2, 2, 2)
        # Index each sample's observed state
        sample_idx = torch.arange(idx.size(0))
        loss = -log_joint[sample_idx, idx[:, 0], idx[:, 1],
                          idx[:, 2], idx[:, 3]].mean()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0 or epoch == N_EPOCHS - 1:
            print(f"  Epoch {epoch:4d}  NLL = {loss.item():.4f}")

    # ── 4. VE queries (averaged over embeddings) vs empirical ─────────
    model.eval()

    def empirical_cond(query_col, query_val, evidence):
        mask = torch.ones(data.size(0), dtype=torch.bool)
        for col, val in evidence.items():
            mask &= data[:, col] == val
        subset = data[mask]
        if subset.size(0) == 0:
            return float('nan')
        return (subset[:, query_col] == query_val).float().mean().item()

    def empirical_joint_cond(query_cols, query_vals, evidence):
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

    print("\n" + "=" * 60)
    print("VE Queries (averaged over embeddings) vs Empirical")
    print("=" * 60)

    def ve_query_avg(query_vars, evidence):
        """Run batched VE and average P(query|x) over matching embeddings."""
        if evidence:
            mask = torch.ones(data.size(0), dtype=torch.bool)
            for k, v in evidence.items():
                mask &= data[:, COL[k]] == v
            embs = embeddings[mask]
        else:
            embs = embeddings
        ev = dict(evidence)
        ev['input'] = embs
        # Get per-concept probabilities, average over batch
        probs = ve.query(query=query_vars, evidence=ev)  # (N, n_features)
        return probs.mean(dim=0)  # average over batch

    with torch.no_grad():
        # Marginals: E_x[ P(var | x) ]
        print("\n--- Marginal probabilities ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        for var in NODE_NAMES:
            avg = ve_query_avg([var], {})
            ve_p = avg.item()
            emp_p = empirical_cond(COL[var], 1, {})
            print(f"  P({var}=1){'':<35s} {ve_p:8.4f}  {emp_p:9.4f}")

        # Forward queries
        print("\n--- Forward queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        for qvar, ev in [("studies", {"economy": 1}),
                         ("studies", {"economy": 1, "talent": 1}),
                         ("job_offer", {"studies": 1})]:
            avg = ve_query_avg([qvar], ev)
            ve_p = avg.item()
            emp_p = empirical_cond(COL[qvar], 1,
                                   {COL[k]: v for k, v in ev.items()})
            ev_str = ", ".join(f"{k}={v}" for k, v in ev.items())
            label = f"P({qvar}=1 | {ev_str})"
            print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")

        # Explaining away
        print("\n--- Explaining-away queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        for qvar, ev in [("economy", {"job_offer": 1}),
                         ("talent", {"job_offer": 1}),
                         ("economy", {"job_offer": 1, "talent": 1})]:
            avg = ve_query_avg([qvar], ev)
            ve_p = avg.item()
            emp_p = empirical_cond(COL[qvar], 1,
                                   {COL[k]: v for k, v in ev.items()})
            ev_str = ", ".join(f"{k}={v}" for k, v in ev.items())
            label = f"P({qvar}=1 | {ev_str})"
            print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")

        # Joint conditional — use return_log_joint for multi-variable joint
        print("\n--- Joint conditional queries ---")
        print(f"  {'query':<45s} {'VE':>8s}  {'Empirical':>9s}")
        ev = {"job_offer": 1}
        mask = data[:, COL["job_offer"]] == 1
        embs = embeddings[mask]
        out = ve.query(query=["economy", "talent"],
                       evidence={'input': embs, "job_offer": 1},
                       return_log_joint=True)
        # Exponentiate log-joint to get P(economy, talent | job_offer=1)
        avg = torch.exp(out['log_joint']).mean(dim=0)  # (2, 2)
        emp_ev = {COL["job_offer"]: 1}
        for e in range(2):
            for t in range(2):
                ve_p = avg[e, t].item()
                emp_p = empirical_joint_cond(
                    [COL["economy"], COL["talent"]], [e, t], emp_ev)
                label = f"P(economy={e}, talent={t} | job_offer=1)"
                print(f"  {label:<45s} {ve_p:8.4f}  {emp_p:9.4f}")


if __name__ == "__main__":
    main()
