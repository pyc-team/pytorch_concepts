"""MAP forward inference: propagating the *most likely* value of every CPD.

We reuse the ASIA (chest-clinic) network from bnlearn::

    asia ──► tub ──┐
                   ├──► either ──┬──► xray
    smoke ─► lung ─┘             │
      │                          │
      └───► bronc ───────────────┴──► dysp

All three forward engines walk the very same topological sweep; they differ
only in what each node hands to its children, and therefore in what lands in
``out.samples``:

- ``DeterministicInference`` propagates the CPD's **parameter** — a Bernoulli's
  probability, e.g. ``0.73``. Soft and differentiable, and what you train with.
  It is not a realisation, so ``out.samples`` stays ``None``.
- ``AncestralSamplingInference`` propagates a **relaxed draw** — a
  reparameterised sample from the Concrete surrogate, e.g. ``0.68``. It fills
  ``out.samples``, but the values are soft (they only approach ``{0, 1}`` as the
  temperature falls) and differ from call to call.
- ``MAPForwardInference`` propagates the CPD's **mode** — a hard ``1.``. Each
  node commits to its single most likely value given its parents' commitments,
  and that decision is what flows downstream. Same input, same output, every
  time.

Because the commitment is per node given its parents, this is a *greedy* MAP
sweep, not the most probable joint assignment (which would need max-product).
It is test-time only and runs under ``torch.no_grad``.

No embeddings are used: ``dataset.input_data`` is ignored, so the model is a
purely concept-to-concept graph.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import ParametricCPD, BayesianNetwork, LearnablePrior, \
    AncestralSamplingInference, DeterministicInference, MAPForwardInference


NAMES = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]


def build_model():
    """One binary concept per ASIA node, each CPD a small MLP over its parents."""
    variables = {
        name: ConceptVariable(name, distribution=Bernoulli, size=1) for name in NAMES
    }
    parents = {
        "asia": [], "smoke": [],
        "tub": ["asia"], "lung": ["smoke"], "bronc": ["smoke"],
        "either": ["tub", "lung"], "xray": ["either"], "dysp": ["bronc", "either"],
    }
    factors = []
    for name, parent_names in parents.items():
        if not parent_names:
            # A root has no parents, so its logit is a bare learnable parameter.
            parametrization = {"logits": LearnablePrior(1)}
        else:
            parametrization = {"logits": nn.Sequential(
                nn.Linear(len(parent_names), 16), nn.ReLU(), nn.Linear(16, 1)
            )}
        factors.append(ParametricCPD(
            variables[name],
            parametrization=parametrization,
            parents=[variables[p] for p in parent_names],
        ))
    return BayesianNetwork(variables=list(variables.values()), factors=factors)


def main():
    seed_everything(42)

    dataset = BnLearnDataset(name="asia", seed=42, n_gen=10000)
    concepts = dataset.concepts.float()          # (N, 8), all binary
    targets = {name: concepts[:, i:i + 1] for i, name in enumerate(NAMES)}

    model = build_model()

    # ---- Train (teacher-forced MLE over the full joint) --------------------
    trainer = AncestralSamplingInference(model, p_int=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    trainer.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = trainer.query(query=dict(targets), evidence={})
        loss = sum(loss_fn(out.logits[name], targets[name]) for name in NAMES)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.3f}")

    # ---- The three forward engines on the same query -----------------------
    # Identical sweep, identical evidence; only the propagated value differs.
    # p_int=0.0 so ancestral propagates its *own* draws rather than the targets.
    engines = {
        "deterministic": DeterministicInference(model),
        "ancestral": AncestralSamplingInference(model, p_int=0.0),
        "map": MAPForwardInference(model),
    }
    for engine in engines.values():
        engine.eval()

    query = ["tub", "lung", "bronc", "either", "xray", "dysp"]
    evidence = {"asia": targets["asia"][:3], "smoke": targets["smoke"][:3]}
    out = {name: engine.query(query=query, evidence=evidence)
           for name, engine in engines.items()}

    fmt = lambda t: " ".join(f"{v:5.2f}" for v in t.squeeze(-1))
    print("\n--- the value each node propagates (first 3 rows) ---")
    print(f"{'':>8} | {'deterministic':^17} | {'ancestral':^17} | {'MAP':^17}")
    print(f"{'node':>8} | {'(probability)':^17} | {'(relaxed draw)':^17} | {'(hard mode)':^17}")
    print("-" * 74)
    for name in query:
        print(f"{name:>8} | {fmt(torch.sigmoid(out['deterministic'].logits[name])):^17} "
              f"| {fmt(out['ancestral'].samples[name]):^17} "
              f"| {fmt(out['map'].samples[name]):^17}")

    # What actually lands in `samples` is the structural difference.
    print("\n--- out.samples ---")
    for name in engines:
        s = out[name].samples
        if s is None:
            print(f"  {name:14s}: None  (propagates a parameter, not a realisation)")
        else:
            hard = ((s == 0.0) | (s == 1.0)).float().mean().item()
            print(f"  {name:14s}: shape {tuple(s.shape)}, {hard:.0%} of values hard 0/1, "
                  f"requires_grad={s.requires_grad}")

    # Same question twice: only the sampler moves.
    print("\n--- stability across two identical calls (max abs difference) ---")
    for name in ("ancestral", "map"):
        again = engines[name].query(query=query, evidence=evidence)
        delta = (again.samples - out[name].samples).abs().max().item()
        print(f"  {name:14s}: {delta:.3f}"
              f"{'   <- stochastic' if delta else '   <- deterministic'}")

    # `params` still carries the raw CPD output each decision came from.
    print(f"\nMAP .logits['dysp'][0] = {out['map'].logits['dysp'][0].item():+.3f} "
          f"-> sample {out['map'].samples['dysp'][0].item():.0f}")

    # ---- Evidence changes the downstream commitment ------------------------
    # `either` is clamped, so `xray`/`dysp` are the modes of *that* branch.
    print("\n--- MAP values with `either` clamped ---")
    one = torch.ones(1, 1)
    for value in (0.0, 1.0):
        clamped = engines["map"].query(
            query=["xray", "dysp"],
            evidence={"asia": one, "smoke": one, "either": one * value},
        )
        print(f"  either={value:.0f} -> xray={clamped.samples['xray'].item():.0f}, "
              f"dysp={clamped.samples['dysp'].item():.0f}")

    # ---- How often the committed value matches the data --------------------
    full = engines["map"].query(
        query=query, evidence={"asia": targets["asia"], "smoke": targets["smoke"]}
    )
    print("\n--- MAP accuracy against the generated data ---")
    for name in query:
        accuracy = (full.samples[name] == targets[name]).float().mean().item()
        print(f"  {name:8s} {accuracy:.3f}")


if __name__ == "__main__":
    main()
