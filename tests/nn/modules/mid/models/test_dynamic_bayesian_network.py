"""DynamicBayesianNetwork: ``previous()``, the initial and transition networks,
the unrolled network, a model of order 2, and the errors of a malformed model.

The model has a window ``x`` at step 0 only; a plate ``c``, a hidden state ``h``
and a binary ``y`` at every step, with one CPD of ``y`` in both lists; and a
plate ``u`` from step 1 on.
"""
import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ConceptVariable, DynamicBayesianNetwork, DynamicInference, EmbeddingVariable, FixedPrior,
    LearnablePrior, ParametricCPD,
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
        assert p._previous_of is v


def test_previous_takes_a_variable_or_a_plate():
    c = _plate("c")
    for handle in (c.previous(), c.member("c_a")):
        with pytest.raises(ValueError, match="previous"):
            handle.previous()


def test_a_member_of_the_step_before_as_a_parent():
    x, c, d = _plate("x"), _plate("c"), ConceptVariable("d", distribution=Delta)
    dbn = DynamicBayesianNetwork(
        variables=[x, c, d],
        initial=[_cpd(x), _cpd(c, x)],
        transition=[_cpd(c, c.previous()), _cpd(d, c.previous().member("c_a"))],
    )
    parent = dbn.transition.factors["d"].parents[0]
    assert parent.name == "c_a_prev" and parent.plate is dbn.transition.variables["c_prev"]
    bn = dbn.unroll(2)
    parent = bn.factors["d_2"].parents[0]
    assert parent.name == "c_a_1" and parent.plate is bn.variables["c_1"]


# ------------------------------------------------- initial and transition ----

def test_initial_and_transition_networks():
    dbn = _model()
    assert dbn.interface == ["c", "h"]
    assert list(dbn.initial[0].variables) == ["x", "c", "h", "y"]
    assert list(dbn.transition.variables) == ["c_prev", "h_prev", "c", "h", "y", "u"]
    assert [p.name for p in dbn.transition.factors["c"].parents] == ["c_prev", "u"]


def test_the_step_before_is_an_observed_root_of_the_transition():
    transition = _model().transition
    c_prev, cpd = transition.variables["c_prev"], transition.factors["c_prev"]
    assert c_prev.members == ["c_a_prev", "c_b_prev"]
    assert type(c_prev) is ConceptVariable and type(transition.variables["h_prev"]) is EmbeddingVariable
    assert c_prev.distribution is Delta
    assert cpd.is_root and isinstance(cpd.parametrization["value"], FixedPrior)


def test_a_stochastic_interface_variable_enters_the_transition_as_a_delta():
    z = ConceptVariable("z", distribution=dist.Bernoulli)
    dbn = DynamicBayesianNetwork(
        variables=[z],
        initial=[ParametricCPD(z, parametrization={"logits": LearnablePrior(1)})],
        transition=[ParametricCPD(z, parents=[z.previous()], parametrization={"logits": nn.Linear(1, 1)})],
    )
    assert dbn.transition.variables["z_prev"].distribution is Delta


def test_the_steps_share_their_modules():
    dbn = _model()
    initial, transition = dbn.initial[0].factors["y"], dbn.transition.factors["y"]
    assert initial.parametrization["logits"] is transition.parametrization["logits"]
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


def test_previous_with_a_lag():
    c = _plate("c")
    p = c.previous(2)
    assert p._previous_of is c and p._lag == 2 and c.previous()._lag == 1
    with pytest.raises(ValueError, match="lag"):
        c.previous(0)


def _order_two():
    """c_0 = e(x); c_1 = f1(c_0); c_t = f2(c_{t-1}, c_{t-2}) for t >= 2."""
    torch.manual_seed(0)
    x, c = EmbeddingVariable("x", distribution=Delta, size=3), _plate("c")
    e, f1, f2 = nn.Linear(3, 2), nn.Linear(2, 2), nn.Linear(4, 2)
    dbn = DynamicBayesianNetwork(
        [x, c],
        initial=[
            [_cpd(x), ParametricCPD(c, parents=[x], parametrization=e)],
            [ParametricCPD(c, parents=[c.previous()], parametrization=f1)],
        ],
        transition=[
            ParametricCPD(c, parents=[c.previous(), c.previous(2)], parametrization=f2)
        ],
    )
    return dbn, (e, f1, f2)


def test_order_two_networks():
    dbn, _ = _order_two()
    assert len(dbn.initial) == 2 and dbn.interface == ["c"]
    assert list(dbn.initial[0].variables) == ["x", "c"]
    assert list(dbn.initial[1].variables) == ["c_prev", "c"]
    assert list(dbn.transition.variables) == ["c_prev", "c_prev2", "c"]
    assert dbn.transition.variables["c_prev2"].members == ["c_a_prev2", "c_b_prev2"]
    assert [p.name for p in dbn.transition.factors["c"].parents] == ["c_prev", "c_prev2"]
    assert dbn.slices("c", 3) == range(0, 4)
    assert dbn.slices("x", 3) == range(0, 1)
    bn = dbn.unroll(3)
    assert [p.name for p in bn.factors["c_1"].parents] == ["c_0"]
    assert [p.name for p in bn.factors["c_2"].parents] == ["c_1", "c_0"]
    assert [p.name for p in bn.factors["c_3"].parents] == ["c_2", "c_1"]
    assert len(list(dbn.parameters())) == 6


def test_order_two_matches_the_recursion():
    dbn, (e, f1, f2) = _order_two()
    xs = torch.randn(5, 3)
    out = DynamicInference(dbn).query(["c"], {"x": xs}, horizon=4).value["c"].tensor
    cs = [e(xs)]
    cs.append(f1(cs[0]))
    for t in range(2, 5):
        cs.append(f2(torch.cat([cs[t - 1], cs[t - 2]], -1)))
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


def test_slices():
    dbn = _model()
    assert dbn.slices("x", 4) == range(0, 1)
    assert dbn.slices("c_a", 4) == range(0, 5)
    assert dbn.slices("y", 4) == range(0, 5)
    assert dbn.slices("u", 4) == range(1, 5)


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
    with pytest.raises(ValueError, match="not the one registered"):
        DynamicBayesianNetwork([c], initial=[_cpd(c)], transition=[_cpd(_plate("c"), c.previous())])


def test_variable_without_a_cpd_raises():
    c, u = _plate("c"), _plate("u")
    with pytest.raises(ValueError, match="no CPD"):
        DynamicBayesianNetwork([c, u], initial=[_cpd(c)], transition=[_cpd(c, c.previous())])


def test_variable_of_step_0_as_a_parent_in_a_transition_cpd_raises():
    x, c = _plate("x"), _plate("c")
    with pytest.raises(ValueError, match="'x' has no CPD in `transition`"):
        DynamicBayesianNetwork([x, c], initial=[_cpd(x), _cpd(c, x)], transition=[_cpd(c, c.previous(), x)])


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


def test_a_name_taken_by_a_copy_of_the_step_before_raises():
    c, c_prev = _plate("c"), _plate("c_prev")
    with pytest.raises(ValueError, match="taken"):
        DynamicBayesianNetwork([c, c_prev], initial=[_cpd(c), _cpd(c_prev)],
                               transition=[_cpd(c, c.previous()), _cpd(c_prev, c.previous())])


def test_a_factor_that_is_not_a_cpd_raises():
    c = _plate("c")
    with pytest.raises(TypeError, match="ParametricCPD"):
        DynamicBayesianNetwork([c], initial=[_cpd(c)], transition=[nn.Linear(2, 2)])
