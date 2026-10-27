"""DynamicInference against the recursion of the model written in plain torch
with the same modules.

The model is the one of ``test_dynamic_bayesian_network.py``: a window ``x`` at
step 0 only; a plate ``c``, a hidden state ``h`` and a binary ``y`` at every
step; a plate ``u`` from step 1 on. A binary chain runs the engines that
condition on the evidence.
"""
import numpy as np
import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.annotations import Annotations
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    AncestralSamplingInference, BeliefPropagation, ConceptVariable, DeterministicInference,
    DynamicBayesianNetwork, DynamicInference, EmbeddingVariable, FixedPrior, LearnablePrior,
    MAPForwardInference, ParametricCPD, PgmpyVariableElimination, RejectionSampling, VariationalInference,
)
from torch_concepts.tensor import AnnotatedTensor

H, B = 4, 5


def _plate(name):
    return ConceptVariable(name, members=[f"{name}_a", f"{name}_b"], distribution=Delta)


def _cpd(variable, *parents):
    """A FixedPrior at a root, a linear map of the parents otherwise."""
    if not parents:
        return ParametricCPD(variable, parametrization=FixedPrior(torch.zeros(variable.size)))
    size = sum(p.size for p in parents)
    return ParametricCPD(variable, parents=list(parents), parametrization=nn.Linear(size, variable.size))


def _model():
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=3)
    c, u = _plate("c"), _plate("u")
    h = EmbeddingVariable("h", distribution=Delta, size=2)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    head = ParametricCPD(y, parents=[c], parametrization={"logits": nn.Linear(2, 1)})
    return DynamicBayesianNetwork(
        variables=[x, c, h, y, u],
        initial=[_cpd(x), _cpd(c, x), _cpd(h, x, c), head],
        transition=[_cpd(u, c.previous()), _cpd(c, c.previous(), u),
                    _cpd(h, c.previous(), c, h.previous()), head],
    )


def _chain():
    """A binary state z and a binary observation o of it at every step."""
    torch.manual_seed(0)
    z = ConceptVariable("z", distribution=dist.Bernoulli)
    o = ConceptVariable("o", distribution=dist.Bernoulli)
    move, emit = nn.Linear(1, 1), nn.Linear(1, 1)
    for m in (move, emit):  # strong links, so that o_2 changes the posterior of z_1
        nn.init.constant_(m.weight, 4.0)
        nn.init.constant_(m.bias, -2.0)
    emission = ParametricCPD(o, parents=[z], parametrization={"logits": emit})
    return DynamicBayesianNetwork(
        variables=[z, o],
        initial=[ParametricCPD(z, parametrization={"logits": LearnablePrior(1)}), emission],
        transition=[ParametricCPD(z, parents=[z.previous()], parametrization={"logits": move}), emission],
    )


def _module(bn, name):
    """The module of the CPD of ``name`` in ``bn``."""
    return next(iter(bn.factors[name].parametrization.values()))


def _reference(dbn, x, horizon, c=None, u=None):
    """The recursion of the model in plain torch, with the modules of ``dbn``.

    ``c`` and ``u`` map a step to the value carried there in place of the
    prediction. Returns the predictions, with time before the event: c, h and
    the logits of y at steps 0..horizon, u at steps 1..horizon.
    """
    c, u = c or {}, u or {}
    first, step = dbn.initial[0], dbn.transition
    c_hat = _module(first, "c")(x)
    c_t = c.get(0, c_hat)
    h = _module(first, "h")(torch.cat([x, c_t], -1))
    steps = [{"c": c_hat, "h": h, "y": _module(first, "y")(c_t)}]
    for t in range(1, horizon + 1):
        u_hat = _module(step, "u")(c_t)
        c_hat = _module(step, "c")(torch.cat([c_t, u.get(t, u_hat)], -1))
        c_next = c.get(t, c_hat)
        h = _module(step, "h")(torch.cat([c_t, c_next, h], -1))
        c_t = c_next
        steps.append({"c": c_hat, "h": h, "y": _module(step, "y")(c_t), "u": u_hat})
    return {n: torch.stack([s[n] for s in steps if n in s], -2) for n in ("c", "h", "y", "u")}


# ------------------------------------------------- against the reference ----

def test_free_forecast_matches_the_reference():
    dbn, x = _model(), torch.randn(B, 3)
    engine = DynamicInference(dbn)
    out = engine.query(["c", "h", "y"], {"x": x}, horizon=H)
    ref = _reference(dbn, x, H)
    assert out.value["c"].shape == (B, H + 1, 2)
    assert out.logits["y"].shape == (B, H + 1, 1)
    assert torch.allclose(out.value["c"].tensor, ref["c"])
    assert torch.allclose(out.value["h"].tensor, ref["h"])
    assert torch.allclose(out.logits["y"].tensor, ref["y"])
    u = engine.query(["u"], {"x": x}, horizon=H).value["u"]  # u exists from step 1 on
    assert u.shape == (B, H, 2)
    assert torch.allclose(u.tensor, ref["u"])


def test_matches_the_engine_on_the_unrolled_network():
    # DynamicInference computes nothing of its own: it translates the keys for an engine on
    # dbn.unroll(H), whose names are c_t, and stacks the steps of that engine's output.
    dbn, x, targets = _model(), torch.randn(B, 3), torch.randn(B, H + 1, 2)
    out = DynamicInference(dbn, p_int=1.0).query({"c": targets, "y": None}, {"x": x}, horizon=H)
    steps = range(H + 1)
    flat = DeterministicInference(dbn.unroll(H), p_int=1.0).query(
        {f"c_{t}": targets[:, t] for t in steps} | {f"y_{t}": None for t in steps}, {"x_0": x}
    )
    assert torch.allclose(out.value["c"].tensor, torch.stack([flat.value[f"c_{t}"].tensor for t in steps], 1))
    assert torch.allclose(out.logits["y"].tensor, torch.stack([flat.logits[f"y_{t}"].tensor for t in steps], 1))

def test_labels_and_groups_are_those_of_the_model():
    dbn, x = _model(), torch.randn(B, 3)
    out = DynamicInference(dbn).query(["c", "y"], {"x": x}, horizon=H)
    assert out.value.annotations.labels == ["c_a", "c_b"]
    assert out.value.annotations.label_groups == {"c": ["c_a", "c_b"]}
    assert out.logits.annotations.labels == ["y"]
    assert out.value["c_a"].shape == (B, H + 1, 1)


def test_teacher_forcing():
    dbn, x, targets = _model(), torch.randn(B, 3), torch.randn(B, H + 1, 2)
    engine = DynamicInference(dbn, p_int=1.0)
    out = engine.query({"c": targets, "y": None}, {"x": x}, horizon=H)
    # The reported c are the predictions, each step made from the targets of the step before.
    ref = _reference(dbn, x, H, c={t: targets[:, t] for t in range(H + 1)})
    assert torch.allclose(out.value["c"].tensor, ref["c"])
    # y reads the forced c.
    assert torch.allclose(out.logits["y"].tensor, _module(dbn.initial[0], "y")(targets))


def test_a_value_at_one_step():
    dbn, x, v = _model(), torch.randn(B, 3), torch.randn(B, 2)
    engine = DynamicInference(dbn, p_int=1.0)
    out = engine.query({"c": None, ("c", 2): v}, {"x": x}, horizon=H)
    assert torch.allclose(out.value["c"].tensor, _reference(dbn, x, H, c={2: v})["c"])


def test_evidence_at_every_step_and_at_one_step():
    dbn, x, v = _model(), torch.randn(B, 3), torch.randn(B, 2)
    engine = DynamicInference(dbn)
    every = engine.query(["c"], {"x": x, "u": torch.zeros(B, H, 2)}, horizon=H)
    one = engine.query(["c"], {"x": x, ("u", 2): v}, horizon=H)
    zeros = {t: torch.zeros(B, 2) for t in range(1, H + 1)}
    assert torch.allclose(every.value["c"].tensor, _reference(dbn, x, H, u=zeros)["c"])
    assert torch.allclose(one.value["c"].tensor, _reference(dbn, x, H, u={2: v})["c"])


def test_member_evidence():
    dbn, x, v = _model(), torch.randn(B, 3), torch.randn(B, 1)
    out = DynamicInference(dbn).query(["c", "y"], {"x": x, ("c_b", 1): v}, horizon=H)
    # Step 1 carries the predicted c_a and the given c_b.
    c_1 = torch.cat([_reference(dbn, x, H)["c"][:, 1, :1], v], -1)
    ref = _reference(dbn, x, H, c={1: c_1})
    assert torch.allclose(out.value["c"].tensor, ref["c"])
    assert torch.allclose(out.logits["y"].tensor, ref["y"])


def test_the_interface_given_at_a_step_without_a_query():
    # Step 1 is fully given and not queried; step 2 reads the given values.
    dbn, x, c_1 = _model(), torch.randn(B, 3), torch.randn(B, 2)
    evidence = {"x": x, ("c", 1): c_1, ("h", 1): torch.randn(B, 2)}
    out = DynamicInference(dbn).query([("y", 0), ("y", 2)], evidence, horizon=H)
    assert torch.allclose(out.logits["y"].tensor, _reference(dbn, x, H, c={1: c_1})["y"][:, [0, 2]])


@pytest.mark.parametrize("lead", [(3,), (2, 3)])
def test_a_step_without_tensors_takes_the_leading_shape_of_the_query(lead):
    # The chain starts from a prior, and its only tensor is an observation at step 1.
    dbn, o = _chain(), torch.ones(*lead, 1)
    out = DynamicInference(dbn).query(["z"], {("o", 1): o}, horizon=2)
    assert out.logits["z"].shape == (*lead, 3, 1)


def test_n_samples_sets_the_batch_of_a_query_without_tensors():
    engine = DynamicInference(_chain())
    assert engine.query(["z"], horizon=2).logits["z"].shape == (1, 3, 1)
    assert engine.query(["z"], horizon=2, n_samples=6).logits["z"].shape == (6, 3, 1)


class _Tril(nn.Module):
    """A Cholesky factor from a linear map: (*leading, n, n)."""

    def __init__(self, size, n):
        super().__init__()
        self.linear, self.n = nn.Linear(size, n * n), n

    def forward(self, x):
        raw = self.linear(x).unflatten(-1, (self.n, self.n)).tril(-1)
        return raw + torch.diag_embed(nn.functional.softplus(self.linear(x)[..., :self.n]) + 0.1)


def test_a_matrix_parameter_keeps_its_two_axes():
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=2)
    z = ConceptVariable("z", distribution=dist.MultivariateNormal, size=2)
    dbn = DynamicBayesianNetwork(
        variables=[x, z],
        initial=[ParametricCPD(x, parametrization=FixedPrior(torch.zeros(2))),
                 ParametricCPD(z, parents=[x], parametrization={"loc": nn.Linear(2, 2), "scale_tril": _Tril(2, 2)})],
        transition=[ParametricCPD(z, parents=[z.previous()],
                                  parametrization={"loc": nn.Linear(2, 2), "scale_tril": _Tril(2, 2)})],
    )
    window = torch.randn(B, 2)
    out = DynamicInference(dbn).query(["z"], {"x": window}, horizon=H)
    first = DeterministicInference(dbn.initial[0]).query(["z"], {"x": window})
    assert out.params["scale_tril"].shape == (B, H + 1, 2, 2)
    assert torch.allclose(out.params["scale_tril"].tensor[:, 0], first.params["scale_tril"].tensor)


def test_temperature_step_keeps_every_engine_at_the_same_temperature():
    o, dbn = torch.ones(3, 1), _chain()
    engine = DynamicInference(dbn, AncestralSamplingInference,
                              annealing="exponential", annealing_rate=0.5, final_temperature=0.05)
    engine.query(["z"], {("o", 1): o}, horizon=2)
    for _ in range(3):
        engine.temperature_step()
    engine.query(["z"], {("o", 1): o}, horizon=4)  # a new engine for a new horizon
    temperatures = {round(float(e.temperature), 6) for e in engine.engines.values()}
    assert len(engine.engines) == 2 and len(temperatures) == 1 and temperatures.pop() < 1.0
    before = [float(e.temperature) for e in engine.engines.values()]
    dbn.eval()
    engine.temperature_step()  # a no-op in eval mode
    assert [float(e.temperature) for e in engine.engines.values()] == before


def test_a_plate_and_one_of_its_members_are_reported_once():
    dbn, x = _model(), torch.randn(B, 3)
    out = DynamicInference(dbn).query(["c", "c_a"], {"x": x}, horizon=H)
    assert out.value.annotations.labels == ["c_a", "c_b"]


def test_gradients_match_the_reference():
    dbn, x = _model(), torch.randn(B, 3)
    params = list(dbn.parameters())
    c = DynamicInference(dbn).query(["c"], {"x": x}, horizon=H).value["c"].tensor
    grads = torch.autograd.grad(c.pow(2).mean(), params, materialize_grads=True)
    ref = torch.autograd.grad(_reference(dbn, x, H)["c"].pow(2).mean(), params, materialize_grads=True)
    assert any(g.abs().sum() > 0 for g in grads)
    for a, b in zip(grads, ref):
        assert torch.allclose(a, b, atol=1e-6)


def test_leading_dims():
    engine = DynamicInference(_model())
    x, u = torch.randn(2, 3, 3), torch.randn(2, 3, H, 2)
    out = engine.query(["c"], {"x": x, "u": u}, horizon=H).value["c"]
    flat = engine.query(["c"], {"x": x.reshape(6, 3), "u": u.reshape(6, H, 2)}, horizon=H).value["c"]
    assert out.shape == (2, 3, H + 1, 2)
    assert torch.allclose(out.tensor.reshape(6, H + 1, 2), flat.tensor)


def test_an_annotated_tensor_is_read_as_its_tensor():
    dbn, x, targets = _model(), torch.randn(B, 3), torch.randn(B, H + 1, 2)
    annotations = Annotations(labels=["c_a", "c_b"], cardinalities=[1, 1], types=["continuous"] * 2)
    engine = DynamicInference(dbn, p_int=1.0)
    out = engine.query({"c": AnnotatedTensor(targets, annotations, axis=-1)}, {"x": x}, horizon=H)
    assert torch.equal(out.value["c"].tensor, engine.query({"c": targets}, {"x": x}, horizon=H).value["c"].tensor)


# ------------------------------------------------------------- engines ----

def test_ancestral_sampling():
    dbn, x = _model(), torch.randn(B, 3)
    engine = DynamicInference(dbn, AncestralSamplingInference, exact=True)
    samples = engine.query(["c", "y"], {"x": x}, horizon=H).samples
    assert samples.shape == (B, H + 1, 3)
    assert samples.annotations.labels == ["c_a", "c_b", "y"]
    assert torch.allclose(samples["c"].tensor, _reference(dbn, x, H)["c"])  # the samples of a Delta are its value


def test_map_forward():
    dbn, x = _model(), torch.randn(B, 3)
    out = DynamicInference(dbn, MAPForwardInference).query(["c", "y"], {"x": x}, horizon=H)
    assert torch.allclose(out.samples["c"].tensor, _reference(dbn, x, H)["c"])
    assert torch.equal(out.samples["y"].tensor, (out.logits["y"].tensor > 0).float())
    # As its query does, the engine builds no graph.
    assert out.samples.tensor.grad_fn is None and all(p.tensor.grad_fn is None for p in out.params.values())


@pytest.mark.parametrize("engine, kwargs", [
    (PgmpyVariableElimination, {}), (BeliefPropagation, {"iters": 10}),
])
def test_an_engine_conditions_on_a_later_step(engine, kwargs):
    dbn, o = _chain(), torch.ones(3, 1)
    out = DynamicInference(dbn, engine, **kwargs).query([("z", 1)], {("o", 2): o}, horizon=2)
    by_hand = engine(dbn.unroll(2), **kwargs).query(["z_1"], evidence={"o_2": o})
    assert out.probs["z"].shape == (3, 1, 1)
    assert torch.allclose(out.probs["z"].tensor[:, 0], by_hand.probs["z_1"].tensor)


def test_variational_inference_reports_the_guide_per_step():
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=2)
    z, o = ConceptVariable("z", distribution=dist.Normal), ConceptVariable("o", distribution=dist.Normal)
    normal = lambda size: {"loc": nn.Linear(size, 1), "scale": nn.Sequential(nn.Linear(size, 1), nn.Softplus())}
    emission = ParametricCPD(o, parents=[z], parametrization=normal(1))
    dbn = DynamicBayesianNetwork(
        variables=[x, z, o],
        initial=[ParametricCPD(x, parametrization=FixedPrior(torch.zeros(2))),
                 ParametricCPD(z, parents=[x], parametrization=normal(2)), emission],
        transition=[ParametricCPD(z, parents=[z.previous()], parametrization=normal(1)), emission],
    )
    guide = normal(1)  # q(z_t | o_t), the same at every step

    def engine(bn):
        latents = {f"z_{t}": ParametricCPD(bn.variables[f"z_{t}"], parents=[bn.variables[f"o_{t}"]],
                                           parametrization=guide) for t in range(3)}
        return VariationalInference(bn, latents=latents)

    observed = torch.randn(B, 3, 1)
    out = DynamicInference(dbn, engine).query(["z"], {"x": torch.randn(B, 2), "o": observed}, horizon=2)
    assert out.guide_params["loc"]["z"].shape == (B, 3, 1)
    assert torch.allclose(out.guide_params["loc"]["z"].tensor, guide["loc"](observed))


def test_rejection_sampling():
    ones = torch.ones(3, 1)
    engine = DynamicInference(_chain(), RejectionSampling, n_samples=200)
    p = engine.query({("z", 1): ones}, {("o", 2): ones}, horizon=2).probabilities
    assert p.shape == (3,)
    assert ((p >= 0) & (p <= 1)).all()


# ------------------------------------------------------------------ errors ----

def test_horizon_must_be_at_least_1():
    with pytest.raises(ValueError, match="horizon"):
        DynamicInference(_model()).query(["c"], {"x": torch.randn(B, 3)}, horizon=0)


@pytest.mark.parametrize("query, evidence, match", [
    (["nope"], {}, "Unknown"),
    ([], {}, "empty"),
    ([("u", 0)], {}, "not at 0"),
    (["c"], {"u": torch.zeros(B, H - 1, 2)}, "stacks them before the event"),  # one step short
    (["y"], {"c_a": torch.zeros(B, H + 1)}, "stacks them before the event"),  # c_a needs (B, H + 1, 1)
    (["c"], {"u": [[0.0, 0.0]] * B}, "must be a Tensor"),
    (["c"], {("u", 2): torch.zeros(B - 1, 2)}, "'u' at step 2 has \\(4,\\)"),
    ([("c", 2)], {("c", 2): torch.zeros(B, 2)}, "both queried and observed"),
    ([("y", 3)], {("c", 2): torch.zeros(B, 2), ("c_a", 2): torch.ones(B, 1)}, "and their plate"),
    (["c", "u"], {}, "same steps"),
])
def test_invalid_queries_raise(query, evidence, match):
    with pytest.raises(ValueError, match=match):
        DynamicInference(_model()).query(query, {"x": torch.randn(B, 3), **evidence}, horizon=H)


@pytest.mark.parametrize("key", [("c", "2"), ("c", True), ("c", 2, 0)])
def test_a_malformed_key_raises(key):
    with pytest.raises(TypeError, match="pair"):
        DynamicInference(_model()).query([key], {"x": torch.randn(B, 3)}, horizon=H)


def test_a_step_may_be_any_integer():
    engine, x = DynamicInference(_model()), torch.randn(B, 3)
    out = engine.query([("c", np.int64(2))], {"x": x}, horizon=H)
    assert torch.equal(out.value["c"].tensor, engine.query([("c", 2)], {"x": x}, horizon=H).value["c"].tensor)


def test_a_static_network_raises():
    with pytest.raises(TypeError, match="DynamicBayesianNetwork"):
        DynamicInference(_model().initial[0])
