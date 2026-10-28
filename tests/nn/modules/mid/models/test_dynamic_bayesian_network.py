"""DynamicBayesianNetwork: ``previous()``, the unrolled network, a model of
order 2, the DBN as the network of its first steps, and the errors of a
malformed model.

The model has a window ``x`` at step 0 only; a plate ``c``, a hidden state ``h``
and a binary ``y`` at every step, with one CPD of ``y`` in both lists; and a
plate ``u`` from step 1 on. A binary plate ``z``, with an observation ``o`` at
every step, runs the engines that unpack the plates.
"""
import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    BayesianNetwork, BeliefPropagation, ConceptVariable, DeterministicInference, DoIntervention,
    DynamicBayesianNetwork, DynamicInference, EmbeddingVariable, FixedPrior, LearnablePrior,
    ParametricCPD, PgmpyVariableElimination, UniformPolicy, intervention, plot_pgm,
)


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


# ---------------------------------------------------------------- previous() ----

def test_previous_is_the_variable_one_step_earlier():
    for v in (_plate("c"), EmbeddingVariable("h", distribution=Delta, shape=(2, 3))):
        p = v.previous()
        assert type(p) is type(v)
        assert (p.name, p.distribution, p.members, p.shape) == (v.name, v.distribution, v.members, v.shape)


def test_previous_takes_a_variable_or_a_plate_and_a_lag_of_at_least_1():
    c = _plate("c")
    for handle in (c.previous(), c.member("c_a")):
        with pytest.raises(ValueError, match="previous"):
            handle.previous()
    with pytest.raises(ValueError, match="lag"):
        c.previous(0)


def test_a_member_of_the_step_before_as_a_parent():
    x, c, d = _plate("x"), _plate("c"), ConceptVariable("d", distribution=Delta)
    dbn = DynamicBayesianNetwork(
        variables=[x, c, d],
        initial=[_cpd(x), _cpd(c, x)],
        transition=[_cpd(c, c.previous()), _cpd(d, c.previous().member("c_a"))],
    )
    bn = dbn.unroll(2)
    parent = bn.factors["d_2"].parents[0]
    assert parent.name == "c_a_1" and parent.plate is bn.variables["c_1"]


def test_each_step_keeps_the_options_of_its_variable():
    z = ConceptVariable("z", distribution=dist.RelaxedBernoulli, dist_kwargs={"temperature": 0.5})
    dbn = DynamicBayesianNetwork(
        variables=[z],
        initial=[ParametricCPD(z, parametrization={"logits": LearnablePrior(1)})],
        transition=[ParametricCPD(z, parents=[z.previous()], parametrization={"logits": nn.Linear(1, 1)})],
    )
    assert dbn.variables["z_1"].dist_kwargs == {"temperature": 0.5}


def test_the_steps_share_their_modules():
    dbn = _model()
    assert dbn.factors["y_0"].parametrization["logits"] is dbn.factors["y_1"].parametrization["logits"]
    assert len(list(dbn.parameters())) == 12  # six linear maps, the head of y counted once


# -------------------------------------------------------- unrolled network ----

def test_unroll():
    dbn = _model()
    bn = dbn.unroll(3)
    assert list(bn.variables) == ["x_0", "c_0", "h_0", "y_0"] + [
        f"{n}_{t}" for t in (1, 2, 3) for n in ("c", "h", "y", "u")]
    assert bn.variables["c_2"].members == ["c_a_2", "c_b_2"]
    assert [p.name for p in bn.factors["c_2"].parents] == ["c_1", "u_2"]
    assert [p.name for p in bn.factors["h_2"].parents] == ["c_1", "c_2", "h_1"]
    assert {id(p) for p in bn.parameters()} == {id(p) for p in dbn.parameters()}
    with pytest.raises(ValueError, match="horizon"):
        dbn.unroll(0)


def _order_two():
    """c_0 = e(x); c_1 = f1(c_0); c_t = f2(trunk(c_{t-1} + c_{t-2})) for t >= 2."""
    torch.manual_seed(0)
    x, c = EmbeddingVariable("x", distribution=Delta, size=3), _plate("c")
    return DynamicBayesianNetwork(
        [x, c],
        initial=[[_cpd(x), _cpd(c, x)], [_cpd(c, c.previous())]],
        transition=[ParametricCPD(  # sums its parents by position: they are renamed at every step
            c, parents=[c.previous(), c.previous(2)], parametrization=nn.Linear(3, 2), trunk=nn.Linear(2, 3),
            aggregate=lambda parents: sum(parents.values()))],
    )


def test_order_two():
    dbn = _order_two()
    assert dbn.steps("c", 3) == range(0, 4)
    assert dbn.steps("x", 3) == range(0, 1)
    bn = dbn.unroll(3)
    assert [p.name for p in bn.factors["c_1"].parents] == ["c_0"]
    assert [p.name for p in bn.factors["c_2"].parents] == ["c_1", "c_0"]
    assert [p.name for p in bn.factors["c_3"].parents] == ["c_2", "c_1"]
    assert {id(p) for p in bn.parameters()} == {id(p) for p in dbn.parameters()}
    assert len(list(dbn.parameters())) == 8


def test_order_two_matches_the_recursion():
    dbn = _order_two()
    e, f1, f2 = (dbn.factors[f"c_{t}"].parametrization["value"] for t in range(3))
    trunk = dbn.factors["c_2"].trunk
    xs = torch.randn(5, 3)
    out = DynamicInference(dbn).query(["c"], {"x": xs}, horizon=4).value["c"].tensor
    cs = [e(xs)]
    cs.append(f1(cs[0]))
    for t in range(2, 5):
        cs.append(f2(trunk(cs[t - 1] + cs[t - 2])))
    assert out.shape == (5, 5, 2) and torch.allclose(out, torch.stack(cs, 1))


def test_order_two_errors():
    x, c = _plate("x"), _plate("c")
    step_0, kernel = [_cpd(x), _cpd(c, x)], [_cpd(c, c.previous())]
    # The transition reads two steps back, but starts at step 1.
    with pytest.raises(ValueError, match="one list per step"):
        DynamicBayesianNetwork([x, c], step_0, [_cpd(c, c.previous(), c.previous(2))])
    # Step 1 reads two steps back.
    with pytest.raises(ValueError, match="at step -1"):
        DynamicBayesianNetwork([x, c], [step_0, [_cpd(c, c.previous(2))]], kernel)
    # Step 1 reads x at step 0, where x has no CPD.
    with pytest.raises(ValueError, match="'x' at step 0"):
        DynamicBayesianNetwork(
            [x, c], [[_cpd(c)], [_cpd(x), _cpd(c, c.previous(), x.previous())]], kernel
        )
    # A CPD at steps 0 and 2, but not at 1.
    with pytest.raises(ValueError, match="consecutive"):
        DynamicBayesianNetwork([x, c], [step_0, [_cpd(x)]], kernel)
    with pytest.raises(ValueError, match=r"more than one CPD in `initial\[1\]`"):
        DynamicBayesianNetwork([x, c], [step_0, [_cpd(c, c.previous()), _cpd(c)]], kernel)


def test_steps():
    # x at step 0 only, c at every step, w at steps 0 and 1 only, v from step 1 on.
    x, c, w, v = EmbeddingVariable("x", distribution=Delta, size=3), _plate("c"), _plate("w"), _plate("v")
    dbn = DynamicBayesianNetwork(
        [x, c, w, v],
        initial=[[_cpd(x), _cpd(c, x), _cpd(w, c)], [_cpd(c, c.previous()), _cpd(w, c), _cpd(v, c)]],
        transition=[_cpd(c, c.previous(), c.previous(2)), _cpd(v, c)],
    )
    assert [dbn.steps(n, 4) for n in ("x", "c", "w", "v", "v_a")] == [
        range(0, 1), range(0, 5), range(0, 2), range(1, 5), range(1, 5)]
    assert dbn.steps("w", 1) == range(0, 2) and dbn.steps("v", 1) == range(1, 2)
    assert list(dbn.unroll(3).variables) == [
        "x_0", "c_0", "w_0", "c_1", "w_1", "v_1", "c_2", "v_2", "c_3", "v_3"]


# ------------------------------------------------ the network of the first steps ----

def _parents(bn):
    return {name: [p.name for p in cpd.parents] for name, cpd in bn.factors.items()}


@pytest.mark.parametrize("model, S, names", [
    (_model, 1, ["x_0", "c_0", "h_0", "y_0", "c_1", "h_1", "y_1", "u_1"]),
    (_order_two, 2, ["x_0", "c_0", "c_1", "c_2"]),  # order 2: steps 0, 1 and 2
])
def test_the_dbn_is_the_network_of_its_first_steps(model, S, names):
    dbn = model()
    first = dbn.unroll(S)
    assert isinstance(dbn, BayesianNetwork)
    assert list(dbn.variables) == list(first.variables) == names
    assert _parents(dbn) == _parents(first)
    assert [v.name for v in dbn.sorted_variables] == [v.name for v in first.sorted_variables]
    # Its names are per step; the names as passed are those of steps() and DynamicInference.
    assert dbn.resolve("c_a_1") is dbn.variables["c_1"] and "c" not in dbn.queryable_names


@pytest.mark.parametrize("model, S", [(_model, 1), (_order_two, 2)])
def test_a_static_engine_on_the_dbn_gives_its_first_steps(model, S):
    dbn, x, steps = model(), torch.randn(5, 3), range(S + 1)
    static = DeterministicInference(dbn).query([f"c_{t}" for t in steps], {"x_0": x})
    dynamic = DynamicInference(dbn).query(["c"], {"x": x}, horizon=S)
    assert torch.allclose(torch.stack([static.value[f"c_{t}"].tensor for t in steps], 1),
                          dynamic.value["c"].tensor)


def test_a_forward_engine_runs_one_step_from_the_steps_before():
    # The evidence on c_1 and c_0 stops the engine there: c_2 is the transition from them.
    dbn = _order_two()
    f2, trunk = dbn.factors["c_2"].parametrization["value"], dbn.factors["c_2"].trunk
    c_1, c_0 = torch.randn(5, 2), torch.randn(5, 2)
    out = DeterministicInference(dbn).query(["c_2"], {"c_1": c_1, "c_0": c_0})
    assert torch.allclose(out.value["c_2"].tensor, f2(trunk(c_1 + c_0)))


def _plated_chain():
    """A binary plate z, of members z_a and z_b, and a binary observation o of it at every step."""
    torch.manual_seed(0)
    z = ConceptVariable("z", members=["z_a", "z_b"], distribution=dist.Bernoulli)
    o = ConceptVariable("o", distribution=dist.Bernoulli)
    emission = ParametricCPD(o, parents=[z], parametrization={"logits": nn.Linear(2, 1)})
    return DynamicBayesianNetwork(
        variables=[z, o],
        initial=[ParametricCPD(z, parametrization={"logits": LearnablePrior(2)}), emission],
        transition=[ParametricCPD(z, parents=[z.previous()], parametrization={"logits": nn.Linear(2, 2)}),
                    emission],
    )


@pytest.mark.parametrize("engine, kwargs", [
    (PgmpyVariableElimination, {}), (BeliefPropagation, {"iters": 10}),
])
def test_an_engine_that_unpacks_the_plates_runs_on_the_dbn(engine, kwargs):
    # Both rebuild the network with one variable per member of a plate.
    dbn, o = _plated_chain(), torch.ones(3, 1)
    static = engine(dbn, **kwargs).query(["z_a_0"], evidence={"o_1": o})
    dynamic = DynamicInference(dbn, engine, **kwargs).query([("z_a", 0)], {("o", 1): o}, horizon=1)
    assert torch.allclose(static.probs["z_a_0"].tensor, dynamic.probs["z_a"].tensor[:, 0])


def test_the_state_dict_is_that_of_the_first_steps():
    dbn = _model()
    assert list(dbn.state_dict()) == list(dbn.unroll(1).state_dict())
    with torch.no_grad():
        for p in dbn.parameters():
            p.add_(1.0)
    fresh = _model()
    fresh.load_state_dict(dbn.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(fresh.parameters(), dbn.parameters()))


@pytest.mark.parametrize("model, name, steps", [
    (_model, "c_0", [0]), (_model, "c_1", [1, 2, 3]),
    (_order_two, "c_1", [1]), (_order_two, "c_2", [2, 3]),
])
def test_an_intervention_acts_at_the_steps_of_its_cpd(model, name, steps):
    # The intervention hooks the module of the CPD that the name has in the network of the first
    # steps; the last of them is the transition, which runs at every later step. DynamicInference
    # is a target through its pgm.
    dbn, x = model(), torch.randn(5, 3)
    engine = DynamicInference(dbn)
    for target in (dbn, engine):
        with intervention(target, DoIntervention(0.0), UniformPolicy(), [name]):
            c = engine.query(["c"], {"x": x}, horizon=3).value["c"].tensor
        assert [t for t in range(4) if (c[:, t] == 0).all()] == steps


def test_an_intervention_on_a_cpd_in_both_lists_acts_at_every_step():
    dbn, x = _model(), torch.randn(5, 3)
    with intervention(dbn, DoIntervention(0.0), UniformPolicy(), ["y_0"]):
        y = DynamicInference(dbn).query(["y"], {"x": x}, horizon=3).logits["y"].tensor
    assert (y == 0).all()


def test_plot_pgm_draws_the_network_of_the_first_steps(tmp_path):
    pytest.importorskip("matplotlib", reason="plot_pgm needs matplotlib")
    path = tmp_path / "dbn.pdf"
    plot_pgm(_model(), path)
    assert path.read_bytes().startswith(b"%PDF")


def test_member_names_that_look_like_steps():
    # The members of b are b_0 and b_1. Per step, b_1 is b at step 1; steps() and the keys of
    # DynamicInference take the names as passed, where b_1 is a member of b.
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=3)
    b = ConceptVariable("b", members=["b_0", "b_1"], distribution=Delta)
    dbn = DynamicBayesianNetwork([x, b], initial=[_cpd(x), _cpd(b, x)], transition=[_cpd(b, b.previous())])
    assert dbn.resolve("b_1") is dbn.variables["b_1"] and dbn.steps("b_1", 3) == range(0, 4)
    engine, xs = DynamicInference(dbn), torch.randn(5, 3)
    whole = engine.query(["b"], {"x": xs}, horizon=3).value["b"].tensor
    member = engine.query({"b_1": torch.zeros(5, 4, 1)}, {"x": xs}, horizon=3).value
    assert member.annotations.labels == ["b_1"] and torch.allclose(member.tensor, whole[..., 1:])
    # b_0 given at step 1, which step 2 reads.
    e, f = (dbn.factors[name].parametrization["value"] for name in ("b_0", "b_1"))
    b_1 = torch.cat([torch.zeros(5, 1), f(e(xs))[:, 1:]], -1)
    out = engine.query([("b", 2)], {"x": xs, ("b_0", 1): torch.zeros(5, 1)}, horizon=3)
    assert torch.allclose(out.value["b"].tensor[:, 0], f(b_1))


# ------------------------------------------------------------------ errors ----

def test_previous_of_a_variable_missing_at_step_0_raises():
    c, u = _plate("c"), _plate("u")
    with pytest.raises(ValueError, match="where it does not exist"):
        DynamicBayesianNetwork([c, u], initial=[_cpd(c)],
                               transition=[_cpd(u, c.previous()), _cpd(c, u.previous())])


def test_previous_parent_in_an_initial_cpd_raises():
    c = _plate("c")
    with pytest.raises(ValueError, match="at step -1"):
        DynamicBayesianNetwork([c], initial=[_cpd(c, c.previous())], transition=[_cpd(c, c.previous())])


def test_cpd_of_another_variable_with_the_same_name_raises():
    c = _plate("c")
    with pytest.raises(ValueError, match="not one of `variables`"):
        DynamicBayesianNetwork([c], initial=[_cpd(c)], transition=[_cpd(_plate("c"), c.previous())])


@pytest.mark.parametrize("parent", [
    lambda: _plate("d"),  # a variable not in `variables`
    lambda: _plate("u"),  # another plate called u
    lambda: _plate("u").member("u_a"),  # a member of it
    lambda: _plate("c").previous(),  # previous() of another plate called c
    lambda: _plate("c").previous().member("c_a"),  # a member of it
    lambda: _plate("d").previous(),  # previous() of a variable not in `variables`
])
def test_a_parent_that_is_not_registered_raises(parent):
    c, u = _plate("c"), _plate("u")
    with pytest.raises(ValueError, match="is not one of `variables`"):
        DynamicBayesianNetwork([c, u], initial=[_cpd(c), _cpd(u)],
                               transition=[_cpd(u), _cpd(c, c.previous(), parent())])


def test_variable_without_a_cpd_raises():
    c, u = _plate("c"), _plate("u")
    with pytest.raises(ValueError, match="no CPD"):
        DynamicBayesianNetwork([c, u], initial=[_cpd(c)], transition=[_cpd(c, c.previous())])


def test_variable_of_step_0_as_a_parent_in_a_transition_cpd_raises():
    x, c = _plate("x"), _plate("c")
    with pytest.raises(ValueError, match=r"reads 'x' at step 1 \(lag 0\)"):
        DynamicBayesianNetwork([x, c], initial=[_cpd(x), _cpd(c, x)], transition=[_cpd(c, c.previous(), x)])
    # From step 2 on, x.previous() reads x at step 1, where x does not exist.
    with pytest.raises(ValueError, match="'x' at step 1"):
        DynamicBayesianNetwork([x, c], initial=[_cpd(x), _cpd(c, x)],
                               transition=[_cpd(c, c.previous(), x.previous())])


def test_two_cpds_of_a_variable_in_one_list_raise():
    c = _plate("c")
    with pytest.raises(ValueError, match="more than one CPD in `transition`"):
        DynamicBayesianNetwork([c], initial=[_cpd(c)], transition=[_cpd(c, c.previous()), _cpd(c, c.previous())])


def test_cycle_within_a_step_raises():
    c, u = _plate("c"), _plate("u")
    with pytest.raises(ValueError, match="cycle"):
        DynamicBayesianNetwork([c, u], initial=[_cpd(c)], transition=[_cpd(u, c), _cpd(c, c.previous(), u)])


@pytest.mark.parametrize("variables, match", [
    ([_plate("c"), _plate("c")], "Duplicate"),
    ([_plate("c"), _plate("c_a")], "must be unique"),  # a variable named as a member of c
])
def test_duplicate_names_raise(variables, match):
    with pytest.raises(ValueError, match=match):
        DynamicBayesianNetwork(variables, initial=[], transition=[])


def test_a_factor_that_is_not_a_cpd_raises():
    c = _plate("c")
    with pytest.raises(TypeError, match="ParametricCPD"):
        DynamicBayesianNetwork([c], initial=[_cpd(c)], transition=[nn.Linear(2, 2)])
