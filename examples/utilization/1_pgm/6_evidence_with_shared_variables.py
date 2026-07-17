"""Evidence and interventions on a plate concept variable.

Model:  x (root, evidence) -> concepts = {c1, c2} (a plate) -> xor   (whole plate)
        concepts member c1 ----------------------------------> y1    (only member c1)

This checks that, with the unified plate design, you can address the group and
its individual members **by name** for both:

- evidence (observe): clamp the whole plate, or a single member (partial
  observation), and have it propagate to the downstream task;
- intervention / do (force a value and propagate): the forward-pass ``do`` —
  set a concept and read the task's response. Member targeting is just the
  member's column of the plate ("the slice does the job"); the group key targets
  the whole thing.

(The richer ``InterventionModule`` API that *replaces a CPD's parametrization*
is unchanged by this refactor and targets the same columns via
``out_concepts_to_intervene_on`` — here we exercise the engine-level forcing,
which is what the plate addressing is responsible for.)
"""

import torch
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD, BayesianNetwork, DeterministicInference,
    LinearEmbeddingToConcept, LinearConceptToConcept, LearnablePrior, Sequential,
)

B, X = 8, 4


def build():
    x = EmbeddingVariable("x", distribution=Delta, size=X)
    concepts = ConceptVariable("concepts", members=["c1", "c2"], distribution=Bernoulli)
    xor = ConceptVariable("xor", distribution=OneHotCategorical, size=2)   # depends on the whole plate
    y1 = ConceptVariable("y1", distribution=Bernoulli)                     # depends on ONLY member c1
    factors = [
        ParametricCPD(x, parents=[], 
            parametrization=LearnablePrior(X)
        ),
        ParametricCPD(concepts, parents=[x], 
            parametrization=Sequential(
                LinearEmbeddingToConcept(in_embeddings=X, out_concepts=concepts.size), 
                torch.nn.Sigmoid()
            )
        ),
        ParametricCPD(xor, parents=[concepts], 
            parametrization=Sequential(
                LinearConceptToConcept(in_concepts=concepts.size, out_concepts=2), 
                torch.nn.Softmax(dim=-1)
            )
        ),
        # child wired to a SINGLE member of the plate via concepts.member("c1")
        ParametricCPD(y1, parents=[concepts.member("c1")], 
            parametrization=Sequential(
                LinearConceptToConcept(in_concepts=1, out_concepts=1), 
                torch.nn.Sigmoid()
            )
        ),
    ]
    return BayesianNetwork(variables=[x, concepts, xor, y1], factors=factors)


def main():
    seed_everything(0)
    model = build()
    engine = DeterministicInference(model)
    x = torch.randn(B, X)
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)

    # --- addressing: group vs member query ---------------------------------
    base = engine.query(["concepts", "xor"], evidence={"x": x})
    c = base.params["concepts"]["probs"]                       # (B, 2) — the whole plate
    members = engine.query(["c1", "c2"], evidence={"x": x})
    assert tuple(c.shape) == (B, 2)
    assert torch.allclose(members.params["c1"]["probs"], c[:, 0:1])   # member = its column (a view)
    assert torch.allclose(members.params["c2"]["probs"], c[:, 1:2])
    print("query: concepts -> (B,2);  c1/c2 -> their columns                 OK")

    # --- evidence: observe the whole plate ---------------------------------
    obs_group = engine.query(["xor"], evidence={"x": x, "concepts": torch.cat([ones, zeros], dim=1)})
    assert not torch.allclose(obs_group.params["xor"]["probs"], base.params["xor"]["probs"])
    print("evidence: observing the whole plate changes xor                   OK")

    # --- evidence: observe ONE member (partial observation) ----------------
    obs_c1 = engine.query(["xor", "c2"], evidence={"x": x, "c1": ones})
    # the unobserved member c2 is still the model's prediction...
    assert torch.allclose(obs_c1.params["c2"]["probs"], c[:, 1:2])
    # ...and clamping c1 propagates to xor.
    assert not torch.allclose(obs_c1.params["xor"]["probs"], base.params["xor"]["probs"])
    print("evidence: observing only c1 leaves c2 as model, moves xor         OK")

    # --- intervention / do: force a member, read the task's response -------
    do_hi = engine.query(["xor"], evidence={"x": x, "c1": ones}).params["xor"]["probs"]
    do_lo = engine.query(["xor"], evidence={"x": x, "c1": zeros}).params["xor"]["probs"]
    assert not torch.allclose(do_hi, do_lo)
    print("do(c1=1) vs do(c1=0): the targeted column changes xor             OK")

    # group-level do: force both members at once
    do_group = engine.query(["xor"], evidence={"x": x, "concepts": torch.cat([ones, ones], dim=1)}).params["xor"]["probs"]
    assert not torch.allclose(do_group, do_lo)
    print("do(concepts=[1,1]) via the group key                              OK")

    # --- child wired to ONLY one member of the plate -----------------------
    # y1's single parent is concepts.member("c1"): forcing c1 moves y1, but
    # forcing c2 (not its parent) leaves y1 unchanged.
    y1_c1 = [engine.query(["y1"], evidence={"x": x, "c1": v}).params["y1"]["probs"] for v in (ones, zeros)]
    y1_c2 = [engine.query(["y1"], evidence={"x": x, "c2": v}).params["y1"]["probs"] for v in (ones, zeros)]
    assert not torch.allclose(y1_c1[0], y1_c1[1])   # c1 is y1's parent -> moves it
    assert torch.allclose(y1_c2[0], y1_c2[1])       # c2 is not -> no effect
    print("subset edge: y1 depends only on c1 -> c1 moves it, c2 does not    OK")


if __name__ == "__main__":
    main()
