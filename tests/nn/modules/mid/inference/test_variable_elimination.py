"""PgmpyVariableElimination: agreement with BP where BP is exact (trees),
exactness where BP is not (frustrated loopy graphs), conditional (CRF)
evidence, leading dims, plate members, and the documented error boundaries."""
import itertools
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.inference.utils import enumerable_cardinality
from torch_concepts.nn.modules.mid.graph.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.graph.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.torch.belief_propagation import (
    BeliefPropagation,
)
from torch_concepts.nn.modules.mid.inference.pgmpy.variable_elimination import (
    PgmpyVariableElimination,
)
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.distributions import Delta


class _ConcreteModel(ProbabilisticModel):
    """Minimal concrete subclass, for mixed graphs the structural subclasses
    would reject."""


# --------------------------------------------------------------------------
# Helpers (mirrored from test_belief_propagation.py)
# --------------------------------------------------------------------------
def _bin(name):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=1)


def _cat(name, k):
    return ConceptVariable(name, distribution=dist.OneHotCategorical, size=k)


def _energy_net(scope, conditioning=None, hidden=16):
    in_dim = sum(v.size for v in scope)
    if conditioning:
        in_dim += sum(v.size for v in conditioning)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def _pot(scope, name, conditioning=None):
    return ParametricPotential(
        scope=list(scope) + list(conditioning or []),
        parametrization=_energy_net(scope, conditioning),
        name=name,
    )


def _pot_mod(scope, module, name):
    """A potential with an explicit (non-MLP) energy module."""
    return ParametricPotential(scope=list(scope), parametrization=module, name=name)


def _encode(v, s, batch):
    card = enumerable_cardinality(v)
    if card == 2 and v.size == 1:
        return torch.full((batch, 1), float(s))
    val = torch.zeros(batch, v.size)
    val[:, s] = 1.0
    return val


def _exact_marginals(fg, free_names, evidence_states=None, conditioning=None, batch=1):
    """Marginals by enumerating the full joint from the factors' log_potential."""
    evidence_states = evidence_states or {}
    free_vars = [fg.variables[n] for n in free_names]
    cards = [enumerable_cardinality(v) for v in free_vars]
    ev_assign = {
        fg.variables[n]: _encode(fg.variables[n], s, batch)
        for n, s in evidence_states.items()
    }
    ev_assign.update({fg.variables[n]: t for n, t in (conditioning or {}).items()})
    scores = []
    for combo in itertools.product(*[range(c) for c in cards]):
        assignment = dict(ev_assign)
        for v, s in zip(free_vars, combo):
            assignment[v] = _encode(v, s, batch)
        tot = 0.0
        for f in fg.factors.values():
            tot = tot + f.log_potential(assignment)
        scores.append(tot)
    logj = torch.stack(scores, dim=-1)
    joint = torch.softmax(logj, dim=-1).reshape(batch, *cards)
    marg = {}
    for i, n in enumerate(free_names):
        axes = tuple(ax for ax in range(1, len(free_names) + 1) if ax != i + 1)
        marg[n] = joint.sum(dim=axes) if axes else joint
    return marg


def _state_marginal(variable, probs):
    """Engine ``probs`` -> ``(batch, cardinality)`` state marginal."""
    if enumerable_cardinality(variable) == 2 and variable.size == 1:
        return torch.cat([1.0 - probs, probs], dim=-1)
    return probs


def _assert_marginals_match(fg, out, exact, names, atol=1e-5):
    for n in names:
        got = _state_marginal(fg.variables[n], out.probs[n])
        assert torch.allclose(got, exact[n], atol=atol), (n, got, exact[n])
        assert torch.allclose(got.sum(-1), torch.ones_like(got.sum(-1)), atol=1e-5)


def _assert_engines_agree(fg, names, evidence=None, atol=1e-6, **bp_kwargs):
    """VE and BP must return the same marginals (only valid where BP is exact)."""
    evidence = evidence or {}
    bp = BeliefPropagation(fg, **{"iters": 25, **bp_kwargs}).query(
        query=names, evidence=evidence
    )
    ve = PgmpyVariableElimination(fg).query(query=names, evidence=evidence)
    for n in names:
        assert torch.allclose(ve.probs[n], bp.probs[n], atol=atol), (
            n, ve.probs[n], bp.probs[n]
        )
    return ve


class _Ising(nn.Module):
    """E([x, y]) = J * (2x-1)(2y-1) — an explicit, frustratable coupling.

    A random MLP energy produces loopy graphs that BP happens to get right to
    ~1e-6, which would make the "VE is exact where BP is not" tests vacuous.
    An explicit antiferromagnetic coupling is what actually frustrates BP.
    """

    def __init__(self, J):
        super().__init__()
        self.J = float(J)

    def forward(self, z):
        s = 2.0 * z - 1.0
        return self.J * s[..., 0:1] * s[..., 1:2]


class _Field(nn.Module):
    """E([x]) = h * (2x-1) — a unary field."""

    def __init__(self, h):
        super().__init__()
        self.h = float(h)

    def forward(self, z):
        return self.h * (2.0 * z - 1.0)


def _frustrated(J=1.5, h=0.5):
    """A 4-cycle with two chords and mixed-sign couplings: loopy and frustrated."""
    a, b, c, d = _bin("a"), _bin("b"), _bin("c"), _bin("d")
    return MarkovNetwork(variables=[a, b, c, d], factors=[
        _pot_mod([a, b], _Ising(+J), "ab"), _pot_mod([b, c], _Ising(+J), "bc"),
        _pot_mod([c, d], _Ising(+J), "cd"), _pot_mod([d, a], _Ising(-J), "da"),
        _pot_mod([a, c], _Ising(-J), "ac"),
        _pot_mod([a], _Field(h), "ha"), _pot_mod([b], _Field(-h), "hb"),
    ])


# --------------------------------------------------------------------------
class TestVEMatchesBPOnTrees:
    """BP is exact on a tree, so any disagreement is a bug in one of the two."""

    def test_chain_matches_bp_and_brute_force(self):
        torch.manual_seed(0)
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        fg = MarkovNetwork(variables=[a, b, c], factors=[
            _pot([a], "ua"), _pot([b], "ub"), _pot([c], "uc"),
            _pot([a, b], "ab"), _pot([b, c], "bc"),
        ])
        names = ["a", "b", "c"]
        out = _assert_engines_agree(fg, names)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_star_matches_bp(self):
        torch.manual_seed(1)
        hub = _cat("h", 3)
        leaves = [_bin(f"l{i}") for i in range(3)]
        factors = [_pot([hub], "uh")]
        for i, leaf in enumerate(leaves):
            factors.append(_pot([hub, leaf], f"e{i}"))
        fg = MarkovNetwork(variables=[hub, *leaves], factors=factors)
        names = ["h", "l0", "l1", "l2"]
        out = _assert_engines_agree(fg, names)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_all_cpd_dag_matches_bp(self):
        """A directed model takes the same DiscreteFactor path: pgmpy never
        sees a TabularCPD and never needs per-parent normalisation."""
        torch.manual_seed(2)
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(
            variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a]
        )
        fg = _ConcreteModel(variables=[a, b], factors=[cpd_a, cpd_b])
        names = ["a", "b"]
        out = _assert_engines_agree(fg, names)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_ragged_cardinalities(self):
        """BP pads a shared state axis, VE does not — agreeing here is a real
        cross-check of both layouts."""
        torch.manual_seed(3)
        a, b, c = _bin("a"), _cat("b", 3), _cat("c", 4)
        fg = MarkovNetwork(variables=[a, b, c], factors=[
            _pot([a, b], "ab"), _pot([b, c], "bc"), _pot([a], "ua"),
        ])
        names = ["a", "b", "c"]
        out = _assert_engines_agree(fg, names)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_unary_only_graph(self):
        """Arity-1 factors add no edges, and two potentials on one node must
        both count."""
        torch.manual_seed(4)
        a, b = _bin("a"), _bin("b")
        fg = MarkovNetwork(variables=[a, b], factors=[
            _pot([a], "ua1"), _pot([a], "ua2"), _pot([b], "ub"),
        ])
        names = ["a", "b"]
        out = _assert_engines_agree(fg, names)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)


class TestVEExactWhereBPIsNot:
    """The point of the engine: a frustrated loopy graph, where BP is wrong."""

    def test_loopy_matches_brute_force(self):
        fg = _frustrated()
        names = ["a", "b", "c", "d"]
        out = PgmpyVariableElimination(fg).query(query=names, evidence={})
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_bp_is_wrong_here(self):
        """Guards the test above from being vacuous: if BP were right too, the
        class would prove nothing about exactness."""
        fg = _frustrated()
        names = ["a", "b", "c", "d"]
        bp = BeliefPropagation(fg, iters=50, damping=0.5).query(
            query=names, evidence={}
        )
        exact = _exact_marginals(fg, names)
        worst = max(
            float((_state_marginal(fg.variables[n], bp.probs[n]) - exact[n]).abs().max())
            for n in names
        )
        assert worst > 0.1, f"BP was accurate to {worst}; the graph is not frustrated"

    def test_extreme_log_potentials(self):
        """Regression for the float64 + max-subtraction stabilisation: at these
        couplings the log-potentials span ~600 nats, so float32 would lose the
        tail to underflow and an unshifted exp would overflow to inf."""
        fg = _frustrated(J=120.0, h=40.0)
        names = ["a", "b", "c", "d"]
        out = PgmpyVariableElimination(fg).query(query=names, evidence={})
        for n in names:
            probs = _state_marginal(fg.variables[n], out.probs[n])
            assert torch.isfinite(probs).all(), (n, probs)
            assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-6)
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)


class TestVEConditionalAndEvidence:
    def test_crf_batched_conditioning(self):
        """Continuous evidence enters through factor reduction — the only route
        it could take, since pgmpy's own ``evidence=`` takes state names."""
        torch.manual_seed(3)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        fg = _ConcreteModel(
            variables=[a, b, emb], factors=[_pot([a, b], "phi", conditioning=[emb])]
        )
        e = torch.randn(7, 4)
        out = PgmpyVariableElimination(fg).query(query=["a", "b"], evidence={"emb": e})
        assert out.probs["a"].shape == (7, 1)
        exact = _exact_marginals(fg, ["a", "b"], conditioning={"emb": e}, batch=7)
        _assert_marginals_match(fg, out, exact, ["a", "b"])

    def test_discrete_evidence_matches_exact_conditional(self):
        torch.manual_seed(4)
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(
            variables=[a, b], factors=[_pot([a], "ua"), _pot([a, b], "ab")]
        )
        out = PgmpyVariableElimination(fg).query(
            query=["a"], evidence={"b": torch.ones(1, 1)}
        )
        _assert_marginals_match(
            fg, out, _exact_marginals(fg, ["a"], evidence_states={"b": 1}), ["a"]
        )

    def test_observed_variable_emits_no_params(self):
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a, b], "ab")])
        out = PgmpyVariableElimination(fg).query(
            query=["a", "b"], evidence={"b": torch.ones(1, 1)}
        )
        assert set(out.variables) == {"a"}

    def test_all_queried_variables_observed(self):
        """Nothing is left to compute: the early return, which BP has no
        equivalent of because it always computes every free node."""
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a, b], "ab")])
        out = PgmpyVariableElimination(fg).query(
            query=["b"], evidence={"b": torch.ones(1, 1)}
        )
        assert out.params == {}


class TestVELeadingDims:
    def test_two_leading_dims_preserved(self):
        torch.manual_seed(6)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=3)
        fg = _ConcreteModel(
            variables=[a, b, emb], factors=[_pot([a, b], "phi", conditioning=[emb])]
        )
        e = torch.randn(2, 5, 3)
        out = PgmpyVariableElimination(fg).query(query=["a", "b"], evidence={"emb": e})
        assert out.probs["a"].shape == (2, 5, 1)
        flat = PgmpyVariableElimination(fg).query(
            query=["a", "b"], evidence={"emb": e.reshape(10, 3)}
        )
        assert torch.allclose(out.probs["a"].reshape(10, 1), flat.probs["a"], atol=1e-6)

    def test_rows_are_independent(self):
        """Rows conditioned identically agree; rows conditioned differently do not."""
        torch.manual_seed(7)
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=3)
        fg = _ConcreteModel(
            variables=[a, emb], factors=[_pot([a], "ua", conditioning=[emb])]
        )
        one = torch.randn(1, 3)
        e = torch.cat([one, one, torch.randn(1, 3) + 5.0], dim=0)
        probs = PgmpyVariableElimination(fg).query(
            query=["a"], evidence={"emb": e}
        ).probs["a"]
        assert torch.allclose(probs[0], probs[1], atol=1e-9)
        assert not torch.allclose(probs[0], probs[2], atol=1e-4)


class TestVEPlateMembers:
    """Plates are erased by ``unpack_plates``, so members are ordinary nodes."""

    def _two_plate_bn(self, seed=30):
        torch.manual_seed(seed)
        root = _bin("root")
        g = ConceptVariable("g", members=["g1", "g2"], distribution=dist.Bernoulli)
        h = ConceptVariable("h", members=["h1", "h2", "h3"], distribution=dist.Bernoulli)
        return BayesianNetwork(
            variables=[root, g, h],
            factors=[
                ParametricCPD(root, {"logits": LearnablePrior(1)}),
                ParametricCPD(g, {"logits": nn.Linear(1, g.size)}, parents=[root]),
                ParametricCPD(h, {"logits": nn.Linear(1, h.size)}, parents=[root]),
            ],
        )

    def _exact_member_marginals(self, fg, evidence_states, batch=1):
        members = [(v, m) for v in fg.variables.values() for m in v.members]
        free = [(v, m) for v, m in members if m not in evidence_states]
        cards = [enumerable_cardinality(v.member(m)) for v, m in free]
        scores = []
        for combo in itertools.product(*[range(c) for c in cards]):
            assign = dict(evidence_states)
            assign.update({m: s for (_, m), s in zip(free, combo)})
            values = {
                v: torch.cat(
                    [_encode(v.member(m), assign[m], batch) for m in v.members], dim=-1
                )
                for v in fg.variables.values()
            }
            total = 0.0
            for f in fg.factors.values():
                total = total + f.log_potential(values)
            scores.append(total)
        joint = torch.softmax(torch.stack(scores, dim=-1), dim=-1).reshape(batch, *cards)
        return {
            m: joint.sum(dim=tuple(ax for ax in range(1, len(free) + 1) if ax != i + 1))
            for i, (_, m) in enumerate(free)
        }

    def _member_probs(self, fg, out, owner_name, member):
        return _state_marginal(fg.variables[owner_name].member(member), out.probs[member])

    def test_no_evidence_matches_exact(self):
        fg = self._two_plate_bn()
        out = PgmpyVariableElimination(fg).query(
            query=["root", "g", "h"], evidence={}
        )
        exact = self._exact_member_marginals(fg, {})
        for owner in ("g", "h"):
            for member in fg.variables[owner].members:
                got = self._member_probs(fg, out, owner, member)
                assert torch.allclose(got, exact[member], atol=1e-5), member

    def test_partial_plate_evidence_matches_exact(self):
        fg = self._two_plate_bn(seed=31)
        one = torch.ones(1, 1)
        out = PgmpyVariableElimination(fg).query(
            query=["root", "g", "h"], evidence={"g1": one}
        )
        exact = self._exact_member_marginals(fg, {"g1": 1})
        for member in ("g2", "h1", "h2", "h3"):
            owner = "g" if member.startswith("g") else "h"
            got = self._member_probs(fg, out, owner, member)
            assert torch.allclose(got, exact[member], atol=1e-5), member
        # An observed member's posterior is a point mass at its evidence.
        assert torch.allclose(out.probs["g1"], one)

    def test_whole_plate_evidence(self):
        fg = self._two_plate_bn(seed=32)
        obs = torch.tensor([[1.0, 0.0]])
        out = PgmpyVariableElimination(fg).query(
            query=["root", "h"], evidence={"g": obs}
        )
        exact = self._exact_member_marginals(fg, {"g1": 1, "g2": 0})
        for member in ("h1", "h2", "h3"):
            got = self._member_probs(fg, out, "h", member)
            assert torch.allclose(got, exact[member], atol=1e-5), member

    def test_single_member_of_a_free_plate(self):
        """Regression: without closing the wanted set over the owning plate,
        ``regroup_members`` raises KeyError on the uncomputed free siblings."""
        fg = self._two_plate_bn(seed=33)
        out = PgmpyVariableElimination(fg).query(query=["h2"], evidence={})
        assert out.probs.annotation.labels == ["h2"]
        exact = self._exact_member_marginals(fg, {})
        got = self._member_probs(fg, out, "h", "h2")
        assert torch.allclose(got, exact["h2"], atol=1e-5)

    def test_matches_bp_on_the_plate_model(self):
        fg = self._two_plate_bn(seed=34)
        one = torch.ones(1, 1)
        names = ["root", "g", "h"]
        bp = BeliefPropagation(fg, iters=30).query(query=names, evidence={"g1": one})
        ve = PgmpyVariableElimination(fg).query(query=names, evidence={"g1": one})
        for member in ("g1", "g2", "h1", "h2", "h3", "root"):
            assert torch.allclose(ve.probs[member], bp.probs[member], atol=1e-5), member


class TestVENoGradAndErrors:
    def test_output_has_no_grad(self):
        """The tables cross into NumPy, so the locked scope is documented by a
        test rather than merely permitted."""
        torch.manual_seed(8)
        a, b = _bin("a"), _bin("b")
        fg = MarkovNetwork(variables=[a, b], factors=[_pot([a, b], "ab")])
        out = PgmpyVariableElimination(fg).query(query=["a"], evidence={})
        assert out.probs["a"].requires_grad is False
        with pytest.raises(RuntimeError):
            out.probs["a"].sum().backward()

    def test_model_parameters_get_no_grad(self):
        torch.manual_seed(9)
        a, b = _bin("a"), _bin("b")
        fg = MarkovNetwork(variables=[a, b], factors=[_pot([a, b], "ab")])
        PgmpyVariableElimination(fg).query(query=["a", "b"], evidence={})
        assert all(p.grad is None for p in fg.parameters())

    def test_continuous_free_variable_raises(self):
        a = _bin("a")
        z = ConceptVariable("z", distribution=Delta, size=2)
        fg = _ConcreteModel(variables=[a, z], factors=[_pot([a, z], "az")])
        with pytest.raises(ValueError, match="not discretely enumerable"):
            PgmpyVariableElimination(fg).query(query=["a"], evidence={})

    def test_non_probabilistic_model_raises(self):
        with pytest.raises(TypeError, match="ProbabilisticModel"):
            PgmpyVariableElimination(nn.Linear(2, 2))

    def test_repr(self):
        a = _bin("a")
        fg = MarkovNetwork(variables=[a], factors=[_pot([a], "ua")])
        assert repr(PgmpyVariableElimination(fg)) == "PgmpyVariableElimination()"


class TestVECost:
    def test_long_chain_stays_cheap(self):
        """Safety net for the joint blowup. pgmpy's default path builds an
        opt_einsum contraction whose output indices are exactly the requested
        variables, i.e. it materialises the *joint* over them: memory
        quadruples per extra binary variable (403 MB at 24), and past ~26 it
        does not even build ("too many subscripts in the output"). Asking one
        variable at a time is linear instead. If someone "simplifies" the
        per-variable loop into a single call, this fails outright rather than
        merely getting slow.
        """
        torch.manual_seed(10)
        n = 40
        variables = [_bin(f"v{i}") for i in range(n)]
        factors = [_pot_mod([v], _Field(0.3), f"u{i}") for i, v in enumerate(variables)]
        factors += [
            _pot_mod([variables[i], variables[i + 1]], _Ising(0.4), f"e{i}")
            for i in range(n - 1)
        ]
        fg = MarkovNetwork(variables=variables, factors=factors)
        names = [v.name for v in variables]
        start = time.perf_counter()
        out = PgmpyVariableElimination(fg).query(query=names, evidence={})
        elapsed = time.perf_counter() - start
        assert set(out.variables) == set(names)
        assert elapsed < 10.0, f"{elapsed:.2f}s — the per-variable loop was lost"

    def test_chain_marginals_are_right(self):
        """The long chain is a tree, so BP is exact on it: a correctness check
        at a size the brute-force helper could never reach."""
        torch.manual_seed(11)
        n = 12
        variables = [_bin(f"v{i}") for i in range(n)]
        factors = [_pot_mod([v], _Field(0.3), f"u{i}") for i, v in enumerate(variables)]
        factors += [
            _pot_mod([variables[i], variables[i + 1]], _Ising(0.4), f"e{i}")
            for i in range(n - 1)
        ]
        fg = MarkovNetwork(variables=variables, factors=factors)
        _assert_engines_agree(fg, [v.name for v in variables], atol=1e-5, iters=60)
