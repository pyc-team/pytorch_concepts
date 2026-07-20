"""Evidence and interventions on a plate concept variable.

Model:  x (root, evidence) -> concepts = {c1, c2} (a plate) -> xor  (whole plate)
        concepts member c1 ---------------------------------> y1   (only member c1)

With the plate design (see ``0.2_cbm_plate.py``) you can address the group and
its individual members **by name** for both:

- evidence (observe): clamp the whole plate, or a single member (partial
  observation), and have it propagate to the downstream task;
- intervention / do (force a value and propagate): set a concept and read the
  task's response. A member is just its column of the plate; the group key
  targets the whole thing.

The example builds a small model, then runs one assertion per property so the
expected behaviour is spelled out in code.
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


def main():
    seed_everything(0)

    # --- model: x -> {c1, c2} (plate) -> xor, plus y1 wired to only c1 -------
    x = EmbeddingVariable("x", distribution=Delta, size=X)
    concepts = ConceptVariable("concepts", members=["c1", "c2"], distribution=Bernoulli)
    xor = ConceptVariable("xor", distribution=OneHotCategorical, size=2)  # from the whole plate
    y1 = ConceptVariable("y1", distribution=Bernoulli)                    # from ONLY member c1
    model = BayesianNetwork(
        variables=[x, concepts, xor, y1],
        factors=[
            ParametricCPD(x, parents=[], parametrization=LearnablePrior(X)),
            ParametricCPD(concepts, parents=[x], parametrization=Sequential(
                LinearEmbeddingToConcept(in_embeddings=X, out_concepts=concepts.size),
                torch.nn.Sigmoid())),
            ParametricCPD(xor, parents=[concepts], parametrization=Sequential(
                LinearConceptToConcept(in_concepts=concepts.size, out_concepts=2),
                torch.nn.Softmax(dim=-1))),
            # child wired to a SINGLE member of the plate via concepts.member("c1")
            ParametricCPD(y1, parents=[concepts.member("c1")], parametrization=Sequential(
                LinearConceptToConcept(in_concepts=1, out_concepts=1),
                torch.nn.Sigmoid())),
        ],
    )

    engine = DeterministicInference(model)
    x_val = torch.randn(B, X)
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)

    # --- addressing: group vs member query ---------------------------------
    base = engine.query(["concepts", "xor"], evidence={"x": x_val})
    c = base.probs["concepts"]                       # (B, 2) — the whole plate
    members = engine.query(["c1", "c2"], evidence={"x": x_val})
    assert tuple(c.shape) == (B, 2)
    assert torch.allclose(members.probs["c1"], c[:, 0:1])   # member = its column (a view)
    assert torch.allclose(members.probs["c2"], c[:, 1:2])
    print("query: concepts -> (B,2);  c1/c2 -> their columns                 OK")

    # --- evidence: observe the whole plate ---------------------------------
    obs_group = engine.query(["xor"], evidence={"x": x_val, "concepts": torch.cat([ones, zeros], dim=1)})
    assert not torch.allclose(obs_group.probs["xor"], base.probs["xor"])
    print("evidence: observing the whole plate changes xor                   OK")

    # --- evidence: observe ONE member (partial observation) ----------------
    obs_c1 = engine.query(["xor", "c2"], evidence={"x": x_val, "c1": ones})
    assert torch.allclose(obs_c1.probs["c2"], c[:, 1:2])    # unobserved c2 still the model's prediction
    assert not torch.allclose(obs_c1.probs["xor"], base.probs["xor"])  # clamping c1 propagates to xor
    print("evidence: observing only c1 leaves c2 as model, moves xor         OK")

    # --- intervention / do: force a member, read the task's response -------
    do_hi = engine.query(["xor"], evidence={"x": x_val, "c1": ones}).probs["xor"]
    do_lo = engine.query(["xor"], evidence={"x": x_val, "c1": zeros}).probs["xor"]
    assert not torch.allclose(do_hi, do_lo)
    print("do(c1=1) vs do(c1=0): the targeted column changes xor             OK")

    # group-level do: force both members at once
    do_group = engine.query(["xor"], evidence={"x": x_val, "concepts": torch.cat([ones, ones], dim=1)}).probs["xor"]
    assert not torch.allclose(do_group, do_lo)
    print("do(concepts=[1,1]) via the group key                              OK")

    # --- child wired to ONLY one member of the plate -----------------------
    # y1's single parent is concepts.member("c1"): forcing c1 moves y1, but
    # forcing c2 (not its parent) leaves y1 unchanged.
    y1_c1 = [engine.query(["y1"], evidence={"x": x_val, "c1": v}).probs["y1"] for v in (ones, zeros)]
    y1_c2 = [engine.query(["y1"], evidence={"x": x_val, "c2": v}).probs["y1"] for v in (ones, zeros)]
    assert not torch.allclose(y1_c1[0], y1_c1[1])   # c1 is y1's parent -> moves it
    assert torch.allclose(y1_c2[0], y1_c2[1])       # c2 is not -> no effect
    print("subset edge: y1 depends only on c1 -> c1 moves it, c2 does not    OK")


if __name__ == "__main__":
    main()
