"""DynamicInference against the recursion of the model written in plain torch
with the same modules, and as an engine: an ``nn.Module`` that holds the DBN,
with one temperature for the engines of every horizon.

The model is the one of ``test_dynamic_bayesian_network.py``: a window ``x`` at
step 0 only; a plate ``c``, a hidden state ``h`` and a binary ``y`` at every
step; a plate ``u`` from step 1 on. A binary chain runs the engines that
condition on the evidence, and those with a temperature schedule.
"""
import math

import numpy as np
import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.annotations import Annotations
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    AncestralSamplingInference, BaseInference, BeliefPropagation, ConceptVariable,
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
    c_hat = _module(dbn, "c_0")(x)  # the CPDs of step 0, then those of the transition
    c_t = c.get(0, c_hat)
    h = _module(dbn, "h_0")(torch.cat([x, c_t], -1))
    steps = [{"c": c_hat, "h": h, "y": _module(dbn, "y_0")(c_t)}]
    for t in range(1, horizon + 1):
        u_hat = _module(dbn, "u_1")(c_t)
        c_hat = _module(dbn, "c_1")(torch.cat([c_t, u.get(t, u_hat)], -1))
        c_next = c.get(t, c_hat)
        h = _module(dbn, "h_1")(torch.cat([c_t, c_next, h], -1))
        c_t = c_next
        steps.append({"c": c_hat, "h": h, "y": _module(dbn, "y_1")(c_t), "u": u_hat})
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


def test_labels_and_groups_are_those_of_the_model():
    dbn, x = _model(), torch.randn(B, 3)
    out = DynamicInference(dbn).query(["c", "c_a", "y"], {"x": x}, horizon=H)  # c_a reported once
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
    assert torch.allclose(out.logits["y"].tensor, _module(dbn, "y_0")(targets))


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
    loc, tril = nn.Linear(2, 2), _Tril(2, 2)  # z at step 0
    dbn = DynamicBayesianNetwork(
        variables=[x, z],
        initial=[ParametricCPD(x, parametrization=FixedPrior(torch.zeros(2))),
                 ParametricCPD(z, parents=[x], parametrization={"loc": loc, "scale_tril": tril})],
        transition=[ParametricCPD(z, parents=[z.previous()],
                                  parametrization={"loc": nn.Linear(2, 2), "scale_tril": _Tril(2, 2)})],
    )
    window = torch.randn(B, 2)
    out = DynamicInference(dbn).query(["z"], {"x": window}, horizon=H)
    assert out.params["scale_tril"].shape == (B, H + 1, 2, 2)
    assert torch.allclose(out.params["scale_tril"].tensor[:, 0], tril(window))


def test_the_time_axis_precedes_an_event_of_two_axes():
    h = EmbeddingVariable("h", distribution=Delta, shape=(2, 3))
    dbn = DynamicBayesianNetwork(
        variables=[h],
        initial=[ParametricCPD(h, parametrization=FixedPrior(torch.zeros(2, 3)))],
        transition=[ParametricCPD(h, parents=[h.previous()], parametrization=nn.Softmax(dim=-1))],
    )
    targets = torch.randn(B, H + 1, 2, 3)
    out = DynamicInference(dbn, p_int=1.0).query({"h": targets}, horizon=H)
    # h_t is the softmax over the last axis of the target of step t - 1; the output, as from a
    # static engine, flattens the event.
    assert torch.allclose(out.value["h"].tensor[:, 1:], targets[:, :-1].softmax(-1).flatten(-2))


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
    out = DynamicInference(dbn, engine, **kwargs).query({("z", 1): o}, {("o", 2): o}, horizon=2)
    by_hand = engine(dbn.unroll(2), **kwargs).query({"z_1": o}, evidence={"o_2": o})
    assert out.probs["z"].shape == (3, 1, 1)
    assert torch.allclose(out.probs["z"].tensor[:, 0], by_hand.probs["z_1"].tensor)
    assert torch.allclose(out.probabilities, by_hand.probabilities)  # P(z_1 = 1 | o_2 = 1)


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


# --------------------------------------------------------------- as an engine ----

def test_dynamic_inference_is_an_engine_on_the_dbn():
    dbn, x = _model(), torch.randn(B, 3)
    engine = DynamicInference(dbn)
    assert isinstance(engine, BaseInference) and isinstance(engine, nn.Module) and engine.pgm is dbn
    assert [id(p) for p in engine.parameters()] == [id(p) for p in dbn.parameters()]
    assert repr(engine) == "DynamicInference(engine=DeterministicInference)" and engine.name == "DynamicInference"
    # called as it is queried, with the horizon
    assert torch.equal(engine(["c"], {"x": x}, horizon=2).value["c"].tensor,
                       engine.query(["c"], {"x": x}, horizon=2).value["c"].tensor)


def test_the_state_dict_holds_the_dbn_and_the_temperature():
    o, schedule = torch.ones(3, 1), dict(annealing="exponential", annealing_rate=0.5, final_temperature=0.05)
    engine = DynamicInference(_chain(), AncestralSamplingInference, **schedule)
    keys = {"_temperature", *(f"pgm.{k}" for k in engine.pgm.state_dict())}
    assert set(engine.state_dict()) == keys
    for horizon in (2, 3):  # the engines of the horizons are not in it
        engine.query(["z"], {("o", 1): o}, horizon=horizon)
    engine.temperature_step()
    assert set(engine.state_dict()) == keys
    fresh = DynamicInference(_chain(), AncestralSamplingInference, **schedule)
    fresh.load_state_dict(engine.state_dict())
    fresh.query(["z"], {("o", 1): o}, horizon=2)
    assert float(fresh._engines[2].temperature) == float(engine.temperature) < 1.0


def test_load_state_dict_with_assign_reaches_an_engine_built_before():
    # load_state_dict(assign=True) replaces the buffer of the temperature, as .to() does.
    o = torch.ones(3, 1)
    engine = DynamicInference(_chain(), AncestralSamplingInference)
    engine.query(["z"], {("o", 1): o}, horizon=2)
    state = engine.state_dict()
    state["_temperature"] = torch.tensor(0.3)
    engine.load_state_dict(state, assign=True)
    engine.query(["z"], {("o", 1): o}, horizon=2)
    assert float(engine._engines[2].temperature) == pytest.approx(0.3)


def test_to_reaches_an_engine_built_before():
    # .to() replaces the buffer of the temperature, which the next query gives to the engine.
    o = torch.ones(3, 1)
    engine = DynamicInference(_chain(), AncestralSamplingInference, annealing="exponential", annealing_rate=0.5)
    engine.query(["z"], {("o", 1): o}, horizon=2)
    engine.to(torch.float64)
    engine.temperature_step()
    out = engine.query(["z"], {("o", 1): o.double()}, horizon=2)
    assert float(engine._engines[2].temperature) == float(engine.temperature) < 1.0
    assert engine._engines[2].temperature.dtype == out.logits["z"].tensor.dtype == torch.float64


def test_clear_cache_drops_the_engines():
    engine, x = DynamicInference(_model()), torch.randn(B, 3)
    before = engine.query(["c"], {"x": x}, horizon=2).value["c"].tensor
    engine.clear_cache()
    assert engine._engines == {}
    assert torch.equal(engine.query(["c"], {"x": x}, horizon=2).value["c"].tensor, before)


def test_the_engines_read_the_schedule_of_dynamic_inference():
    # A function that takes the options of DynamicInference reads its schedule.
    passed = DynamicInference(_chain(), lambda bn, **kw: AncestralSamplingInference(bn, **kw),
                              initial_temperature=0.5)
    passed.query(["z"], {("o", 1): torch.ones(3, 1)}, horizon=2)
    assert float(passed._engines[2].temperature) == float(passed.temperature) == 0.5


@pytest.mark.parametrize("own", [dict(initial_temperature=0.5), dict(annealing="exponential"),
                                 dict(annealing_rate=0.5), dict(final_temperature=0.05)])
def test_an_engine_with_a_schedule_of_its_own_raises(own):
    engine = DynamicInference(_chain(), lambda bn: AncestralSamplingInference(bn, **own))
    for _ in range(2):  # at every query: the engine is not kept
        with pytest.raises(ValueError, match="pass the schedule to DynamicInference"):
            engine.query(["z"], {("o", 1): torch.ones(3, 1)}, horizon=2)


def test_the_schedule_steps_every_engine_and_reaches_its_floor():
    o = torch.ones(3, 1)
    engine = DynamicInference(_chain(), AncestralSamplingInference,
                              annealing="exponential", annealing_rate=0.5, final_temperature=0.05)
    engine.query(["z"], {("o", 1): o}, horizon=2)
    engine.temperature_step()
    engine.query(["z"], {("o", 1): o}, horizon=4)  # a second engine, then one more step
    engine.temperature_step()
    assert [float(e.temperature) for e in engine._engines.values()] == pytest.approx([math.exp(-1.0)] * 2)
    for _ in range(10):
        engine.temperature_step()
    assert float(engine.temperature) == pytest.approx(0.05)


def test_the_constructors_draw_no_random_numbers():
    # so that building a model does not change the random numbers of what follows
    c = _plate("c")
    initial, transition = [_cpd(c)], [_cpd(c, c.previous())]
    state = torch.get_rng_state()
    DynamicInference(DynamicBayesianNetwork([c], initial, transition))
    assert torch.equal(torch.get_rng_state(), state)


# ------------------------------------------------------------------ errors ----

def test_horizon_must_be_at_least_1():
    with pytest.raises(ValueError, match="horizon"):
        DynamicInference(_model()).query(["c"], {"x": torch.randn(B, 3)}, horizon=0)


@pytest.mark.parametrize("query, evidence, match", [
    ([], {}, "empty"),
    ([("u", 0)], {}, "not at 0"),
    (["c"], {"u": torch.zeros(B, H - 1, 2)}, "stacks them before the event"),  # one step short
    (["y"], {"c_a": torch.zeros(B, H + 1)}, "stacks them before the event"),  # c_a needs (B, H + 1, 1)
    (["c"], {"u": [[0.0, 0.0]] * B}, "must be a Tensor"),
    ({("c", 2): [[0.0, 0.0]] * B}, {}, "must be a Tensor"),  # the target of a pair too
    (["c"], {("u", 2): torch.zeros(B - 1, 2)}, "mismatched leading .*'u_2': \\(4,\\)"),  # the engine's check
    ([("c", 2)], {("c", 2): torch.zeros(B, 2)}, "both queried and observed"),
    ([("y", 3)], {("c", 2): torch.zeros(B, 2), ("c_a", 2): torch.ones(B, 1)}, "and their plate"),
    (["c", "u"], {}, "same steps"),
    # the names listed are those of the model, not those of the network of its first steps
    ([("nope", 2)], {}, r"Unknown query name 'nope'; the names are \['c', 'c_a', 'c_b', 'h', 'u'"),
    (["c"], {("nope", 1): torch.zeros(B, 2)}, "Unknown evidence name 'nope'"),
    (["c_2"], {}, "Unknown query name 'c_2'"),  # c at step 2 is ("c", 2)
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
        DynamicInference(_model().unroll(1))


def test_p_int_is_checked_at_construction():
    with pytest.raises(ValueError, match="p_int"):
        DynamicInference(_model(), p_int=2.0)
    assert DynamicInference(_model(), p_int=1.0).p_int == 1.0


def test_steps_that_report_different_parameters_raise():
    # z is parametrized by its logits at step 0 and by its probabilities after.
    z = ConceptVariable("z", distribution=dist.Bernoulli)
    dbn = DynamicBayesianNetwork(
        variables=[z],
        initial=[ParametricCPD(z, parametrization={"logits": LearnablePrior(1)})],
        transition=[ParametricCPD(z, parents=[z.previous()],
                                  parametrization={"probs": nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())})],
    )
    engine = DynamicInference(dbn)
    with pytest.raises(ValueError, match="same parameters at every step"):
        engine.query(["z"], horizon=2)
    # Queried separately, as the message says.
    assert engine.query([("z", 0)], horizon=2).logits["z"].shape == (1, 1, 1)
    assert engine.query([("z", 1), ("z", 2)], horizon=2).probs["z"].shape == (1, 2, 1)
