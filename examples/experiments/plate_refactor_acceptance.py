"""Acceptance + scaling benchmark for the plate refactor (see
PLATE_REFACTOR_INSTRUCTIONS.md §11).

Phase 1 verifies the full plate contract (addressing, evidence, member-handle
parents, owner-keyed CPD forward, extract helpers, uniform engine support).
Phase 2 benchmarks query latency vs. number of plate members.
"""
import statistics
import time
import warnings

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts import seed_everything, ConceptVariable, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD, BayesianNetwork, LearnablePrior,
    DeterministicInference, AncestralSamplingInference,
    RejectionSampling, ImportanceSampling, MutilatedNetworkProposal,
)

X = 16  # root embedding size


# ---------------------------------------------------------------- builders
def build_verification_model():
    """x -> concepts{c1,c2,c3} -> y ; y1 depends on member c1 only."""
    x = EmbeddingVariable("x", distribution=Delta, size=X)
    concepts = ConceptVariable(
        "concepts", members=["c1", "c2", "c3"], distribution=Bernoulli
    )
    y = ConceptVariable("y", distribution=Bernoulli)
    y1 = ConceptVariable("y1", distribution=Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(X)),
        ParametricCPD(
            concepts, parents=[x],
            parametrization=nn.Sequential(nn.Linear(X, concepts.size), nn.Sigmoid()),
        ),
        ParametricCPD(
            y, parents=[concepts],
            parametrization=nn.Sequential(nn.Linear(concepts.size, 1), nn.Sigmoid()),
        ),
        ParametricCPD(
            y1, parents=[concepts.member("c1")],
            parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()),
        ),
    ]
    return BayesianNetwork(variables=[x, concepts, y, y1], factors=factors)


def build_plate_model(n):
    """x -> plate of n Bernoulli members -> y."""
    x = EmbeddingVariable("x", distribution=Delta, size=X)
    concepts = ConceptVariable(
        "concepts", members=[f"c{i}" for i in range(n)], distribution=Bernoulli
    )
    y = ConceptVariable("y", distribution=Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(X)),
        ParametricCPD(
            concepts, parents=[x],
            parametrization=nn.Sequential(nn.Linear(X, n), nn.Sigmoid()),
        ),
        ParametricCPD(
            y, parents=[concepts],
            parametrization=nn.Sequential(nn.Linear(n, 1), nn.Sigmoid()),
        ),
    ]
    return BayesianNetwork(variables=[x, concepts, y], factors=factors)


def build_separate_model(n):
    """Same graph, but one independent variable + CPD per concept (baseline)."""
    x = EmbeddingVariable("x", distribution=Delta, size=X)
    cs = [ConceptVariable(f"c{i}", distribution=Bernoulli) for i in range(n)]
    y = ConceptVariable("y", distribution=Bernoulli)
    factors = [ParametricCPD(x, parametrization=LearnablePrior(X))]
    factors += [
        ParametricCPD(
            c, parents=[x],
            parametrization=nn.Sequential(nn.Linear(X, 1), nn.Sigmoid()),
        )
        for c in cs
    ]
    factors.append(
        ParametricCPD(
            y, parents=cs,
            parametrization=nn.Sequential(nn.Linear(n, 1), nn.Sigmoid()),
        )
    )
    return BayesianNetwork(variables=[x, *cs, y], factors=factors)


# ---------------------------------------------------------------- phase 1
def verify():
    seed_everything(7)
    B = 8
    pgm = build_verification_model()
    eng = DeterministicInference(pgm)
    xt = torch.randn(B, X)
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)

    # 1. group vs member addressing (existing behavior)
    base = eng.query(["concepts", "y", "y1"], evidence={"x": xt})
    c = base.params["concepts"]["probs"]
    assert c.shape == (B, 3), f"plate params shape {tuple(c.shape)} != {(B, 3)}"
    mem = eng.query(["c1", "c2", "c3"], evidence={"x": xt})
    for i, name in enumerate(["c1", "c2", "c3"]):
        assert torch.allclose(mem.params[name]["probs"], c[:, i:i + 1]), \
            f"member {name} != its plate column"
    print("1. group/member query addressing                          OK")

    # 2. whole-plate evidence propagates (existing behavior)
    obs = eng.query(["y"], evidence={"x": xt, "concepts": torch.cat([ones, zeros, ones], dim=1)})
    assert not torch.allclose(obs.params["y"]["probs"], base.params["y"]["probs"])
    print("2. whole-plate evidence moves the task                    OK")

    # 3. partial member evidence: c1 forced, c2/c3 untouched, y moves (existing)
    part = eng.query(["y", "c2", "c3"], evidence={"x": xt, "c1": ones})
    assert torch.allclose(part.params["c2"]["probs"], c[:, 1:2])
    assert torch.allclose(part.params["c3"]["probs"], c[:, 2:3])
    assert not torch.allclose(part.params["y"]["probs"], base.params["y"]["probs"])
    print("3. partial (member) evidence = forcing                    OK")

    # 4. member-handle parent: c1 moves y1, c2 does not (existing behavior)
    y1_c1 = [eng.query(["y1"], evidence={"x": xt, "c1": v}).params["y1"]["probs"] for v in (ones, zeros)]
    y1_c2 = [eng.query(["y1"], evidence={"x": xt, "c2": v}).params["y1"]["probs"] for v in (ones, zeros)]
    assert not torch.allclose(y1_c1[0], y1_c1[1])
    assert torch.allclose(y1_c2[0], y1_c2[1])
    print("4. member-handle parent edge (y1 <- c1 only)              OK")

    # 5. ancestral engine: member samples are plate columns (existing behavior)
    anc = AncestralSamplingInference(pgm)
    s = anc.query(["concepts", "c2", "y"], evidence={"x": xt})
    assert s.samples["concepts"].shape == (B, 3)
    assert torch.allclose(s.samples["c2"], s.samples["concepts"][:, 1:2])
    print("5. ancestral sampling member exposure                     OK")

    # 6. (post-refactor) owner-keyed CPD forward, exact keys still accepted
    c_val = torch.rand(B, 3)
    cpd_y1 = pgm.factors["y1"]
    p_owner = cpd_y1(parent_values={"concepts": c_val})      # owner key (new)
    p_exact = cpd_y1(parent_values={"c1": c_val[:, 0:1]})    # exact key (old contract)
    assert torch.allclose(p_owner["probs"], p_exact["probs"])
    superset = cpd_y1(parent_values={"concepts": c_val, "x": xt})  # extra keys ignored
    assert torch.allclose(superset["probs"], p_owner["probs"])
    print("6. ParametricCPD.forward resolves owner-keyed parents     OK")

    # 7. (post-refactor) uniform read primitives
    vals = {"concepts": c_val}
    assert torch.allclose(pgm.extract("c2", vals), c_val[:, 1:2])
    assert torch.allclose(pgm.extract("concepts", vals), c_val)
    prm = {"concepts": {"probs": c_val}}
    assert torch.allclose(pgm.extract_params("c3", prm)["probs"], c_val[:, 2:3])
    print("7. ProbabilisticModel.extract / extract_params            OK")

    # 8. (post-refactor) member names in the sampling engines. Evidence goes on
    # member c1, not on the Delta root x (see §11 preamble).
    B4 = 4
    ones4 = torch.ones(B4, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rej = RejectionSampling(pgm, n_samples=500)
        p = rej.query({"y": ones4}, evidence={"c1": ones4}).probabilities
        assert p.shape == (B4,) and torch.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        imp = ImportanceSampling(
            pgm, proposal=MutilatedNetworkProposal(pgm), n_samples=200
        )
        p = imp.query({"c2": ones4}, evidence={"c1": ones4}).probabilities
        assert p.shape == (B4,) and torch.isfinite(p).all()
    print("8. member names in Rejection / ImportanceSampling         OK")

    # 9. (post-refactor) Pyro engines, skipped if pyro is absent
    try:
        import pyro  # noqa: F401
        from torch_concepts.nn import VariationalInference, PyroImportanceSampling
        vi = VariationalInference(pgm)
        seed_everything(0)
        y_hi = vi.query(query=["y"], evidence={"x": xt, "c1": ones}).params["y"]["probs"]
        seed_everything(0)
        y_lo = vi.query(query=["y"], evidence={"x": xt, "c1": zeros}).params["y"]["probs"]
        assert not torch.allclose(y_hi, y_lo), "member evidence ignored by model_fn"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pis = PyroImportanceSampling(pgm, n_samples=200)
            p = pis.query({"y": ones4}, evidence={"c1": ones4}).probabilities
            assert p.shape == (B4,) and torch.isfinite(p).all()
        print("9. Pyro engines honor member evidence                     OK")
    except ImportError:
        print("9. pyro-ppl not installed                                 SKIPPED")

    # 10. plates are parameter-agnostic: a Normal plate slices EVERY parameter
    # (loc AND scale) per member — nothing assumes probs/logits.
    x2 = EmbeddingVariable("x", distribution=Delta, size=X)
    lat = ConceptVariable("lat", members=["m1", "m2"], distribution=Normal)
    z = ConceptVariable("z", distribution=Bernoulli)
    npgm = BayesianNetwork(
        variables=[x2, lat, z],
        factors=[
            ParametricCPD(x2, parametrization=LearnablePrior(X)),
            ParametricCPD(lat, parents=[x2], parametrization={
                "loc": nn.Linear(X, lat.size),
                "scale": nn.Sequential(nn.Linear(X, lat.size), nn.Softplus()),
            }),
            ParametricCPD(z, parents=[lat.member("m1")],
                          parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())),
        ],
    )
    neng = DeterministicInference(npgm)
    nout = neng.query(["lat", "m1", "m2", "z"], evidence={"x": xt})
    for param in ("loc", "scale"):
        assert nout.params["lat"][param].shape == (B, 2)
        assert torch.allclose(nout.params["m1"][param], nout.params["lat"][param][:, 0:1])
        assert torch.allclose(nout.params["m2"][param], nout.params["lat"][param][:, 1:2])
    # member evidence on a Normal member forces z through the member edge
    z_hi = neng.query(["z"], evidence={"x": xt, "m1": ones}).params["z"]["probs"]
    z_lo = neng.query(["z"], evidence={"x": xt, "m1": -ones}).params["z"]["probs"]
    assert not torch.allclose(z_hi, z_lo)
    # ancestral sampling draws the whole plate with reparameterised Normals
    ns = AncestralSamplingInference(npgm).query(["lat", "m2"], evidence={"x": xt})
    assert ns.samples["lat"].shape == (B, 2)
    assert torch.allclose(ns.samples["m2"], ns.samples["lat"][:, 1:2])
    print("10. Normal (loc/scale) plate: all params sliced uniformly OK")

    print("\nALL CHECKS PASSED — running the scaling benchmark.\n")


# ---------------------------------------------------------------- phase 2
def _median_ms(engine, query, evidence, repeats=10):
    with torch.no_grad():
        engine.query(query, evidence=evidence)  # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            engine.query(query, evidence=evidence)
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def bench(sweep=(2, 10, 50, 100, 500, 1000), batch=256):
    seed_everything(0)
    header = (
        f"{'n members':>10} | {'plate group (ms)':>17} | "
        f"{'plate members (ms)':>19} | {'separate vars (ms)':>19}"
    )
    print(header)
    print("-" * len(header))
    for n in sweep:
        xt = torch.randn(batch, X)
        members = [f"c{i}" for i in range(n)]

        plate_eng = DeterministicInference(build_plate_model(n))
        t_group = _median_ms(plate_eng, ["concepts", "y"], {"x": xt})
        t_members = _median_ms(plate_eng, members + ["y"], {"x": xt})

        sep_eng = DeterministicInference(build_separate_model(n))
        t_sep = _median_ms(sep_eng, members + ["y"], {"x": xt})

        print(f"{n:>10} | {t_group:>17.2f} | {t_members:>19.2f} | {t_sep:>19.2f}")

    print(
        "\nExpected: 'plate group' ~flat in n; 'plate members' adds only O(n)\n"
        "output slicing; 'separate vars' grows ~linearly (n CPD forwards).\n"
        "A 'plate group' column growing with n means per-member work leaked\n"
        "back into the engine loop."
    )


if __name__ == "__main__":
    verify()
    bench()
