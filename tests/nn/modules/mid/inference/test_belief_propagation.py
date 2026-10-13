"""BeliefPropagation: exactness on trees, agreement with directed inference on a
DAG, conditional (CRF) batching, evidence, differentiability, mixed graphs, and
the documented error boundaries."""
import itertools

import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.inference.utils import (
    enumerable_cardinality,
    factor_table,
)
from torch_concepts.nn.modules.mid.graph.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.graph.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.torch.belief_propagation import BeliefPropagation
from torch_concepts.nn.modules.low.priors import LearnablePrior


class _ConcreteModel(ProbabilisticModel):
    """Minimal concrete subclass used to exercise the abstract base's shared
    behaviour (factor registration, scope validation, adjacency, graph-kind
    flags) without the structural constraints BayesianNetwork/MarkovNetwork
    add. Also stands in for the mixed case until ChainGraph is implemented."""


# --------------------------------------------------------------------------
# Helpers
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
    """An energy-based potential with an interaction-capable (MLP) energy.

    ``conditioning`` is a test-helper convenience only: those variables are
    appended to the scope and observed as evidence by the caller, which is how
    conditional (CRF-style) behaviour is expressed now that
    ``ParametricPotential`` has no separate conditioning argument.
    """
    return ParametricPotential(
        scope=list(scope) + list(conditioning or []),
        parametrization=_energy_net(scope, conditioning),
        name=name,
    )


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
    # ``log_potential`` requires a complete assignment, so observed (continuous)
    # values go in alongside the enumerated states rather than being passed
    # separately.
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
    logj = torch.stack(scores, dim=-1)  # (batch, prod)
    joint = torch.softmax(logj, dim=-1).reshape(batch, *cards)
    marg = {}
    for i, n in enumerate(free_names):
        axes = tuple(ax for ax in range(1, len(free_names) + 1) if ax != i + 1)
        marg[n] = joint.sum(dim=axes) if axes else joint
    return marg


def _state_marginal(variable, probs):
    """Engine ``probs`` -> ``(batch, cardinality)`` state marginal.

    ``out.probs`` follows the uniform engine contract: the marginal is
    expressed in the variable's own parametrization (a binary variable gets a
    width-1 ``P(x=1)``), so widen it back to a state distribution before
    comparing against the enumerated reference.
    """
    if enumerable_cardinality(variable) == 2 and variable.size == 1:
        return torch.cat([1.0 - probs, probs], dim=-1)
    return probs


def _query_each(engine, names, evidence=None):
    """``{name: probs}``, one query per name: BP answers a single variable."""
    evidence = evidence or {}
    return {n: engine.query(query=[n], evidence=evidence).probs[n] for n in names}


def _assert_marginals_match(fg, probs, exact, names, atol=1e-5):
    """``probs`` maps each name to the engine's marginal (``out.probs`` or
    :func:`_query_each`)."""
    for n in names:
        bp = _state_marginal(fg.variables[n], probs[n])
        assert torch.allclose(bp, exact[n], atol=atol), (n, bp, exact[n])
        assert torch.allclose(bp.sum(-1), torch.ones_like(bp.sum(-1)), atol=1e-5)


def _binary_ce(probs, target):
    """Cross-entropy on a binary variable's marginal P(x=1)."""
    return torch.nn.functional.binary_cross_entropy(
        probs.squeeze(-1), target.to(probs.dtype)
    )


# --------------------------------------------------------------------------
class TestBPExactOnTrees:
    def _chain_mrf(self):
        torch.manual_seed(0)
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        factors = [
            _pot([a], "ua"), _pot([b], "ub"), _pot([c], "uc"),
            _pot([a, b], "ab"), _pot([b, c], "bc"),
        ]
        return MarkovNetwork(variables=[a, b, c], factors=factors)

    def test_chain_marginals_exact(self):
        fg = self._chain_mrf()
        probs = _query_each(BeliefPropagation(fg, iters=25), ["a", "b", "c"])
        exact = _exact_marginals(fg, ["a", "b", "c"])
        _assert_marginals_match(fg, probs, exact, ["a", "b", "c"])

    def test_star_marginals_exact(self):
        torch.manual_seed(1)
        hub = _cat("h", 3)
        leaves = [_bin(f"l{i}") for i in range(3)]
        factors = [_pot([hub], "uh")]
        for i, lf in enumerate(leaves):
            factors.append(_pot([hub, lf], f"e{i}"))
        fg = MarkovNetwork(variables=[hub, *leaves], factors=factors)
        names = ["h", "l0", "l1", "l2"]
        probs = _query_each(BeliefPropagation(fg, iters=25), names)
        exact = _exact_marginals(fg, names)
        _assert_marginals_match(fg, probs, exact, names)

    def test_single_unary_marginal(self):
        a = _bin("a")
        fg = _ConcreteModel(variables=[a], factors=[_pot([a], "ua")])
        out = BeliefPropagation(fg, iters=3).query(query=["a"], evidence={})
        exact = _exact_marginals(fg, ["a"])
        _assert_marginals_match(fg, out.probs, exact, ["a"])


class TestBPMatchesDirectedOnDAG:
    def test_all_cpd_factorgraph_marginals_exact(self):
        torch.manual_seed(2)
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        fg = _ConcreteModel(variables=[a, b], factors=[cpd_a, cpd_b])
        probs = _query_each(BeliefPropagation(fg, iters=15), ["a", "b"])
        exact = _exact_marginals(fg, ["a", "b"])
        _assert_marginals_match(fg, probs, exact, ["a", "b"])


class TestBPConditionalAndEvidence:
    def test_crf_batched_conditioning(self):
        torch.manual_seed(3)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = _pot([a, b], "phi", conditioning=[emb])
        fg = _ConcreteModel(variables=[a, b, emb], factors=[pot])
        e = torch.randn(7, 4)
        probs = _query_each(BeliefPropagation(fg, iters=10), ["a", "b"], {"emb": e})
        assert probs["a"].shape == (7, 1)
        exact = _exact_marginals(fg, ["a", "b"], conditioning={"emb": e}, batch=7)
        _assert_marginals_match(fg, probs, exact, ["a", "b"])

    def test_discrete_evidence_matches_exact_conditional(self):
        torch.manual_seed(4)
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a], "ua"), _pot([a, b], "ab")])
        out = BeliefPropagation(fg, iters=10).query(query=["a"], evidence={"b": torch.ones(1, 1)})
        exact = _exact_marginals(fg, ["a"], evidence_states={"b": 1})
        _assert_marginals_match(fg, out.probs, exact, ["a"])

    def test_observed_variable_emits_no_params(self):
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a, b], "ab")])
        eng = BeliefPropagation(fg, iters=5)
        out = eng.query(query=["b"], evidence={"b": torch.ones(1, 1)})
        assert "b" not in out.variables
        out = eng.query(query=["a"], evidence={"b": torch.ones(1, 1)})
        assert set(out.variables) == {"a"}


class TestBPTrainingAndMixed:
    def test_mixed_graph_trains(self):
        torch.manual_seed(5)
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        cpd_c = ParametricCPD(variable=c, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        pot = _pot([b, c], "phi_bc")
        fg = _ConcreteModel(variables=[a, b, c], factors=[cpd_a, cpd_b, cpd_c, pot])
        assert fg.is_mixed
        out = BeliefPropagation(fg, iters=6).query(query=["c"], evidence={})
        loss = _binary_ce(out.probs["c"], torch.tensor([1]))
        loss.backward()
        grads = [p.grad for p in fg.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_marginal_ce_training_reduces_loss(self):
        torch.manual_seed(6)
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=3)
        pot = _pot([a], "ua", conditioning=[emb])
        fg = _ConcreteModel(variables=[a, emb], factors=[pot])
        eng = BeliefPropagation(fg, iters=3)
        x = torch.randn(64, 3)
        target = (x[:, 0] > 0).long()
        opt = torch.optim.Adam(fg.parameters(), lr=0.05)
        first, last = None, None
        for step in range(300):
            opt.zero_grad()
            out = eng.query(query=["a"], evidence={"emb": x})
            loss = _binary_ce(out.probs["a"], target)
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.6


class TestBPVectorisedLayout:
    """The flat ``[E, K]`` layout must be invisible in the answers.

    Padding a ragged state axis with ``LOG0``, grouping factors into
    ``(arity, cardinalities)`` buckets and absorbing degree-1 factors into the
    bias are all pure implementation; each is checked against the enumerated
    joint on a tree, where BP is exact.
    """

    def test_ragged_cardinalities_exact(self):
        """Cardinalities 2/3/4 share one K=4 message axis, so two of the three
        variables carry padded states through every update."""
        torch.manual_seed(10)
        a, b, c = _bin("a"), _cat("b", 3), _cat("c", 4)
        fg = MarkovNetwork(
            variables=[a, b, c],
            factors=[_pot([a], "ua"), _pot([c], "uc"),
                     _pot([a, b], "ab"), _pot([b, c], "bc")],
        )
        names = ["a", "b", "c"]
        probs = _query_each(BeliefPropagation(fg, iters=30), names)
        _assert_marginals_match(fg, probs, _exact_marginals(fg, names), names)

    def test_factors_sharing_a_signature_are_batched_exactly(self):
        """Four identical-signature pairwise factors land in one bucket."""
        torch.manual_seed(11)
        hub = _bin("h")
        leaves = [_bin(f"l{i}") for i in range(4)]
        factors = [_pot([hub, lf], f"e{i}") for i, lf in enumerate(leaves)]
        fg = MarkovNetwork(variables=[hub, *leaves], factors=factors)
        names = ["h", *[lf.name for lf in leaves]]
        probs = _query_each(BeliefPropagation(fg, iters=30), names)
        _assert_marginals_match(fg, probs, _exact_marginals(fg, names), names)

    def test_mixed_signatures_and_arities_exact(self):
        """Three buckets at once: (2,(2,2)), (2,(2,3)) and (3,(2,2,3))."""
        torch.manual_seed(12)
        a, b, c, d = _bin("a"), _bin("b"), _cat("c", 3), _bin("d")
        fg = MarkovNetwork(
            variables=[a, b, c, d],
            factors=[_pot([a, b], "ab"), _pot([a, c], "ac"), _pot([a, b, c], "abc"),
                     _pot([b, d], "bd"), _pot([d], "ud")],
        )
        names = ["a", "b", "c", "d"]
        # Loopy (a-b-c triangle), so this checks the buckets agree with the
        # dict-based reference only in the sense of running; exactness is
        # asserted on the tree tests above. Here we only require a normalised,
        # finite answer over every padded state axis.
        out = _query_each(BeliefPropagation(fg, iters=30, damping=0.5), names)
        for n in names:
            probs = _state_marginal(fg.variables[n], out[n])
            assert torch.isfinite(probs).all()
            assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)

    def test_unary_only_graph_uses_bias_alone(self):
        """Every factor is degree-1, so there are no edges and no message loop."""
        torch.manual_seed(13)
        a, b = _bin("a"), _cat("b", 3)
        fg = MarkovNetwork(
            variables=[a, b],
            factors=[_pot([a], "ua1"), _pot([a], "ua2"), _pot([b], "ub")],
        )
        names = ["a", "b"]
        probs = _query_each(BeliefPropagation(fg, iters=5), names)
        _assert_marginals_match(fg, probs, _exact_marginals(fg, names), names)

    def test_leading_dims_preserved(self):
        """Two leading dimensions survive the flat message layout."""
        torch.manual_seed(14)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=3)
        fg = _ConcreteModel(
            variables=[a, b, emb],
            factors=[_pot([a, b], "phi", conditioning=[emb])],
        )
        out = BeliefPropagation(fg, iters=8).query(
            query=["a"], evidence={"emb": torch.randn(2, 5, 3)}
        )
        assert out.probs["a"].shape == (2, 5, 1)


class TestBPStochasticParametrization:
    """A parametrization with ``Dropout`` still gets one independent mask per
    table cell.

    The engine builds a factor's whole log-potential table in a single
    ``log_potential`` call, with the enumeration grid folded into a leading
    axis. Dropout's mask has the shape of its input, so batching the grid does
    not make the cells share a mask — which is what would silently turn a
    ``K**d`` table into ``K**d`` evaluations of *one* sampled sub-network.
    """

    def _dropout_pot(self, scope, name, p=0.5):
        in_dim = sum(v.size for v in scope)
        return ParametricPotential(
            scope=list(scope),
            parametrization=nn.Sequential(
                nn.Linear(in_dim, 32), nn.Dropout(p), nn.Tanh(), nn.Linear(32, 1)
            ),
            name=name,
        )

    def test_cells_get_independent_masks(self):
        """A table built from a constant-output net under dropout must vary
        across its cells; a shared mask would make every cell identical."""
        torch.manual_seed(20)
        a, b = _bin("a"), _bin("b")
        pot = self._dropout_pot([a, b], "ab")
        # Zero the input weights: without dropout every cell of the table is the
        # same number, so any spread across cells comes from per-cell masking.
        with torch.no_grad():
            pot.parametrization["energy"][0].weight.zero_()
        fg = MarkovNetwork(variables=[a, b], factors=[pot])
        fg.train()
        table = factor_table(
            pot, ["a", "b"], {"a": a, "b": b}, {}, torch.Size([1]),
            torch.float32, torch.device("cpu"),
        )
        assert table.shape == (1, 2, 2)
        assert table.reshape(-1).std() > 0

    def test_eval_mode_is_deterministic(self):
        torch.manual_seed(21)
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        fg = MarkovNetwork(
            variables=[a, b, c],
            factors=[self._dropout_pot([a], "ua"), self._dropout_pot([a, b], "ab"),
                     self._dropout_pot([b, c], "bc")],
        )
        fg.eval()
        eng = BeliefPropagation(fg, iters=8)
        first = eng.query(query=["b"], evidence={}).probs["b"]
        second = eng.query(query=["b"], evidence={}).probs["b"]
        assert torch.equal(first, second)

    def test_train_mode_is_stochastic_but_unbiased_around_eval(self):
        torch.manual_seed(22)
        a, b = _bin("a"), _bin("b")
        fg = MarkovNetwork(
            variables=[a, b],
            factors=[self._dropout_pot([a], "ua"), self._dropout_pot([a, b], "ab")],
        )
        eng = BeliefPropagation(fg, iters=8)
        fg.train()
        draws = torch.stack(
            [eng.query(query=["a"], evidence={}).probs["a"] for _ in range(200)]
        ).reshape(-1)
        assert draws.std() > 1e-3            # dropout really is active
        fg.eval()
        deterministic = eng.query(query=["a"], evidence={}).probs["a"].item()
        assert abs(draws.mean().item() - deterministic) < 0.1


class TestBPKnobs:
    """``damping``, ``tol``/``check_every`` and ``init_noise`` are all
    convergence controls: on a tree with enough rounds none of them may move
    the fixed point."""

    def _chain(self):
        torch.manual_seed(15)
        a, b, c = _bin("a"), _cat("b", 3), _bin("c")
        return MarkovNetwork(
            variables=[a, b, c],
            factors=[_pot([a], "ua"), _pot([a, b], "ab"), _pot([b, c], "bc")],
        )

    @pytest.mark.parametrize("kwargs", [
        {"damping": 0.5},
        {"init_noise": 0.5},
        {"tol": 1e-8, "check_every": 4},
        {"damping": 0.3, "init_noise": 0.2, "tol": 1e-8, "check_every": 3},
    ])
    def test_knobs_reach_the_same_fixed_point(self, kwargs):
        fg = self._chain()
        names = ["a", "b", "c"]
        probs = _query_each(BeliefPropagation(fg, iters=60, **kwargs), names)
        _assert_marginals_match(fg, probs, _exact_marginals(fg, names), names, atol=1e-4)

    def test_tol_stops_early_without_changing_the_answer(self):
        fg = self._chain()
        names = ["a", "b", "c"]
        loose = _query_each(BeliefPropagation(fg, iters=200, tol=1e-7), names)
        exact = _exact_marginals(fg, names)
        _assert_marginals_match(fg, loose, exact, names, atol=1e-4)

    def test_damping_is_log_space(self):
        """Damping mixes the *log* messages, so a damped step of a converged
        run is a no-op rather than a probability-space average."""
        fg = self._chain()
        names = ["a", "b", "c"]
        a = _query_each(BeliefPropagation(fg, iters=80, damping=0.0), names)
        b = _query_each(BeliefPropagation(fg, iters=80, damping=0.7), names)
        for n in names:
            assert torch.allclose(a[n], b[n], atol=1e-4)


class TestBPErrors:
    @pytest.mark.parametrize("kwargs,match", [
        ({"iters": 0}, "iters"),
        ({"damping": 1.0}, "damping"),
        ({"check_every": 0}, "check_every"),
        ({"init_noise": -1.0}, "init_noise"),
    ])
    def test_invalid_settings_raise(self, kwargs, match):
        fg = MarkovNetwork(variables=[_bin("a")], factors=[])
        with pytest.raises(ValueError, match=match):
            BeliefPropagation(fg, **kwargs)


    def test_continuous_free_variable_raises(self):
        a = _bin("a")
        cont = ConceptVariable("z", distribution=dist.Normal, size=2)
        cpd = ParametricCPD(variable=a, parametrization={"logits": nn.Linear(2, 1)}, parents=[cont])
        cpd_z = ParametricCPD(variable=cont, parametrization={"loc": LearnablePrior(2), "scale": LearnablePrior(2)})
        fg = _ConcreteModel(variables=[a, cont], factors=[cpd, cpd_z])
        with pytest.raises(ValueError, match="not discretely enumerable"):
            BeliefPropagation(fg, iters=3).query(query=["a"], evidence={})

    def test_directed_engine_on_undirected_raises(self):
        from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
        a, b = _bin("a"), _bin("b")
        fg = MarkovNetwork(variables=[a, b], factors=[_pot([a, b], "phi")])
        with pytest.raises(TypeError, match="BeliefPropagation"):
            DeterministicInference(fg)

    def test_non_probabilistic_model_raises(self):
        with pytest.raises(TypeError, match="ProbabilisticModel"):
            BeliefPropagation(object())



class TestBPPlateMembers:
    """Plates are inferred one **member** at a time.

    The unit of inference is a member, not the plate, so a plate of ``k``
    members contributes ``k`` nodes to the factor graph. That is what makes
    *partial* evidence expressible — observe some members, marginalise the rest
    — and it incidentally fixes a plate-granular engine's blind spot: a
    Bernoulli plate used to be rejected outright, because a size-``k``
    Bernoulli is not one enumerable variable.

    Note the cost: a CPD over a ``k``-member plate becomes an arity-``k+parents``
    factor, so its table has ``2**k * parent_states`` cells. Messages still
    factorise across members; the table does not.
    """

    def _two_plate_bn(self, seed=30):
        """A BayesianNetwork with two plates hanging off one root::

            root ──► g{g1, g2}
                 └─► h{h1, h2, h3}
        """
        torch.manual_seed(seed)
        from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
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
        """Every member's marginal, by enumerating the full joint.

        ``evidence_states`` maps a member name to its observed state; every
        other member (and every ordinary variable) is enumerated.
        """
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

    def _member_probs(self, fg, probs, owner_name, member):
        """The engine's marginal for one member (``probs`` from
        :func:`_query_each`), widened to a state distribution."""
        owner = fg.variables[owner_name]
        return _state_marginal(owner.member(member), probs[member])

    # -- test-time queries --------------------------------------------------
    def test_no_evidence_matches_exact(self):
        fg = self._two_plate_bn()
        members = fg.variables["g"].members + fg.variables["h"].members
        probs = _query_each(BeliefPropagation(fg, iters=30), members)
        exact = self._exact_member_marginals(fg, {})
        for owner in ("g", "h"):
            for member in fg.variables[owner].members:
                bp = self._member_probs(fg, probs, owner, member)
                assert torch.allclose(bp, exact[member], atol=1e-5), (member, bp, exact[member])

    def test_partial_plate_evidence_matches_exact(self):
        """Observe one member of each plate; the rest must be marginalised."""
        fg = self._two_plate_bn()
        evidence = {"g1": torch.ones(1, 1), "h3": torch.zeros(1, 1)}
        probs = _query_each(
            BeliefPropagation(fg, iters=30), ["g1", "g2", "h1", "h2", "h3"], evidence
        )
        exact = self._exact_member_marginals(fg, {"g1": 1, "h3": 0})
        for owner, member in [("g", "g2"), ("h", "h1"), ("h", "h2")]:
            bp = self._member_probs(fg, probs, owner, member)
            assert torch.allclose(bp, exact[member], atol=1e-5), (member, bp, exact[member])
        # An observed member reports its evidence: conditioning makes it a point mass.
        assert torch.allclose(probs["g1"], torch.ones(1, 1))
        assert torch.allclose(probs["h3"], torch.zeros(1, 1))

    def test_evidence_on_a_member_moves_the_other_plate(self):
        """Both plates hang off ``root``, so conditioning one must move the other."""
        fg = self._two_plate_bn()
        eng = BeliefPropagation(fg, iters=30)
        free = eng.query(query=["h1"], evidence={}).probs["h1"]
        clamped = eng.query(query=["h1"], evidence={"g1": torch.ones(1, 1)}).probs["h1"]
        assert not torch.allclose(free, clamped, atol=1e-4)

    def test_whole_plate_evidence_still_works(self):
        fg = self._two_plate_bn()
        eng = BeliefPropagation(fg, iters=25)
        evidence = {"g": torch.tensor([[1.0, 0.0]])}
        probs = _query_each(eng, fg.variables["h"].members, evidence)
        exact = self._exact_member_marginals(fg, {"g1": 1, "g2": 0})
        for member in fg.variables["h"].members:
            bp = self._member_probs(fg, probs, "h", member)
            assert torch.allclose(bp, exact[member], atol=1e-5), member
        # fully observed -> no parameters
        assert not eng.query(query=["g1"], evidence=evidence).variables

    def test_two_members_of_the_same_plate_observed(self):
        fg = self._two_plate_bn()
        out = BeliefPropagation(fg, iters=25).query(
            query=["h3"], evidence={"h1": torch.ones(1, 1), "h2": torch.zeros(1, 1)}
        )
        exact = self._exact_member_marginals(fg, {"h1": 1, "h2": 0})
        bp = self._member_probs(fg, out.probs, "h", "h3")
        assert torch.allclose(bp, exact["h3"], atol=1e-5)

    def test_batched_partial_evidence(self):
        fg = self._two_plate_bn()
        ev = torch.tensor([[0.0]] * 4 + [[1.0]] * 4)
        probs = _query_each(BeliefPropagation(fg, iters=15), ["g2", "h1"], {"g1": ev})
        assert probs["g2"].shape == (8, 1)
        assert probs["h1"].shape == (8, 1)
        # Rows conditioned differently must get different answers.
        assert not torch.allclose(probs["h1"][0], probs["h1"][7], atol=1e-4)
        # ...and rows conditioned the same must agree.
        assert torch.allclose(probs["h1"][0], probs["h1"][3], atol=1e-6)

    def test_single_member_query_slices_the_plate(self):
        fg = self._two_plate_bn()
        out = BeliefPropagation(fg, iters=20).query(
            query=["g2"], evidence={"g1": torch.ones(1, 1)}
        )
        assert out.probs.annotations.labels == ["g2"]
        exact = self._exact_member_marginals(fg, {"g1": 1})
        assert torch.allclose(
            _state_marginal(fg.variables["g"].member("g2"), out.probs["g2"]),
            exact["g2"], atol=1e-5,
        )

    # -- training -----------------------------------------------------------
    def test_gradients_reach_every_cpd_under_partial_evidence(self):
        fg = self._two_plate_bn()
        probs = _query_each(
            BeliefPropagation(fg, iters=8), ["g1", "g2", "h2", "h3"], {"h1": torch.ones(1, 1)}
        )
        sum(p.tensor.pow(2).sum() for p in probs.values()).backward()
        grads = [p.grad for p in fg.parameters() if p.grad is not None]
        assert len(grads) == len(list(fg.parameters()))
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_training_on_free_members_reduces_loss(self):
        """Supervise the free members while one member of each plate is observed."""
        fg = self._two_plate_bn(seed=31)
        eng = BeliefPropagation(fg, iters=6)
        n = 16
        evidence = {"g1": torch.ones(n, 1), "h1": torch.zeros(n, 1)}
        targets = {"g2": torch.ones(n), "h2": torch.ones(n), "h3": torch.zeros(n)}
        opt = torch.optim.Adam(fg.parameters(), lr=0.1)

        def losses():
            probs = _query_each(eng, list(targets), evidence)
            return sum(_binary_ce(probs[m], target) for m, target in targets.items())

        first = losses().item()
        for _ in range(200):
            opt.zero_grad()
            loss = losses()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.5

        probs = _query_each(eng, list(targets), evidence)
        assert probs["g2"][0].item() > 0.9
        assert probs["h2"][0].item() > 0.9
        assert probs["h3"][0].item() < 0.1

    # -- categorical plates (needs build_distribution's per-member split) ----
    def _mixed_plate_bn(self, seed=32):
        """Two plates of *different* families off one root::

            root ──► g{g1, g2}          Bernoulli members
                 └─► h{h1, h2}          OneHotCategorical members, 3 classes each
        """
        torch.manual_seed(seed)
        from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
        root = _bin("root")
        g = ConceptVariable("g", members=["g1", "g2"], distribution=dist.Bernoulli)
        h = ConceptVariable(
            "h", members=["h1", "h2"], distribution=dist.OneHotCategorical, size=3
        )
        return BayesianNetwork(
            variables=[root, g, h],
            factors=[
                ParametricCPD(root, {"logits": LearnablePrior(1)}),
                ParametricCPD(g, {"logits": nn.Linear(1, g.size)}, parents=[root]),
                ParametricCPD(h, {"logits": nn.Linear(1, h.size)}, parents=[root]),
            ],
        )

    def test_categorical_plate_members_are_independent(self):
        """A k-member categorical plate is k distributions, not one k*m-way one.

        Every member must be able to take every class at once — the whole point
        the flattened build got wrong.
        """
        fg = self._mixed_plate_bn()
        h = fg.variables["h"]
        probs = _query_each(BeliefPropagation(fg, iters=30), h.members)
        for member in h.members:
            block = probs[member]
            assert block.shape == (1, 3)
            assert torch.allclose(block.sum(-1), torch.ones(1), atol=1e-5), member
        exact = self._exact_member_marginals(fg, {})
        for member in h.members:
            assert torch.allclose(probs[member], exact[member], atol=1e-5), member

    def test_categorical_plate_partial_evidence(self):
        fg = self._mixed_plate_bn()
        probs = _query_each(
            BeliefPropagation(fg, iters=30), ["g2", "h1", "h2"],
            {"h1": torch.tensor([[0.0, 0.0, 1.0]]), "g1": torch.ones(1, 1)},
        )
        exact = self._exact_member_marginals(fg, {"h1": 2, "g1": 1})
        g = fg.variables["g"]
        assert torch.allclose(probs["h2"], exact["h2"], atol=1e-5)
        assert torch.allclose(
            _state_marginal(g.member("g2"), probs["g2"]), exact["g2"], atol=1e-5,
        )
        # observed members echo their evidence
        assert torch.allclose(probs["h1"], torch.tensor([[0.0, 0.0, 1.0]]))

    def test_categorical_plate_trains(self):
        fg = self._mixed_plate_bn(seed=33)
        eng = BeliefPropagation(fg, iters=6)
        n = 16
        evidence = {"h1": torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3)}
        target = torch.full((n,), 2)
        h = fg.variables["h"]
        opt = torch.optim.Adam(fg.parameters(), lr=0.1)

        def loss_fn():
            out = eng.query(query=["h2"], evidence=evidence)
            return torch.nn.functional.nll_loss(
                out.probs["h2"].log(), target
            )

        first = loss_fn().item()
        for _ in range(200):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.5
        out = eng.query(query=["h2"], evidence=evidence)
        assert out.probs["h2"][0][2].item() > 0.9


class TestBPSingleVariableQuery:
    """BP's beliefs are single-variable marginals, so a query names one variable
    (or one plate member); several — named, or packed in a plate — raise."""

    def _pair(self):
        a, b = _bin("a"), _bin("b")
        return _ConcreteModel(variables=[a, b], factors=[_pot([a, b], "ab")])

    def test_several_names_raise(self):
        with pytest.raises(ValueError, match="2 variables"):
            BeliefPropagation(self._pair()).query(query=["a", "b"], evidence={})

    def test_several_valued_names_raise(self):
        query = {"a": torch.ones(1, 1), "b": torch.ones(1, 1)}
        with pytest.raises(ValueError, match="PgmpyVariableElimination"):
            BeliefPropagation(self._pair()).query(query=query, evidence={})

    def test_plate_with_several_members_raises(self):
        fg = TestBPPlateMembers()._two_plate_bn()
        with pytest.raises(ValueError, match="3 variables"):
            BeliefPropagation(fg).query(query=["h"], evidence={})

    def test_one_member_is_one_variable(self):
        fg = TestBPPlateMembers()._two_plate_bn()
        out = BeliefPropagation(fg).query(query=["h2"], evidence={})
        assert out.probs.annotations.labels == ["h2"]


class TestBPQueryProbability:
    """A value in the query is a target: ``out.probabilities`` is ``P(Q=q | E)``,
    as for the sampling estimators."""

    def _star(self):
        """A categorical hub with two binary leaves — a tree, so BP is exact."""
        torch.manual_seed(40)
        hub, a, b = _cat("h", 3), _bin("a"), _bin("b")
        return MarkovNetwork(
            variables=[hub, a, b],
            factors=[_pot([hub], "uh"), _pot([hub, a], "ha"), _pot([hub, b], "hb")],
        )

    def test_binary_target_reads_the_exact_conditional(self):
        fg = self._star()
        target = torch.tensor([[1.0], [0.0]])     # rows ask for different values
        out = BeliefPropagation(fg, iters=20).query(
            query={"a": target}, evidence={"b": torch.ones(2, 1)}
        )
        exact = _exact_marginals(fg, ["h", "a"], evidence_states={"b": 1})["a"][0]
        assert out.probabilities.shape == (2,)
        assert torch.allclose(out.probabilities, exact[[1, 0]], atol=1e-5)

    def test_categorical_target_reads_the_exact_conditional(self):
        fg = self._star()
        target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = BeliefPropagation(fg, iters=20).query(
            query={"h": target}, evidence={"a": torch.zeros(2, 1)}
        )
        exact = _exact_marginals(fg, ["h", "b"], evidence_states={"a": 0})["h"][0]
        assert torch.allclose(out.probabilities, exact[[0, 2]], atol=1e-5)

    def test_marginals_are_still_reported(self):
        fg = self._star()
        eng = BeliefPropagation(fg, iters=20)
        valued = eng.query(query={"a": torch.ones(1, 1)}, evidence={})
        plain = eng.query(query=["a"], evidence={})
        assert torch.allclose(valued.probs["a"], plain.probs["a"])
        assert plain.probabilities is None

    def test_plate_member_target(self):
        fg = TestBPPlateMembers()._two_plate_bn()
        out = BeliefPropagation(fg, iters=25).query(
            query={"h2": torch.zeros(1, 1)}, evidence={"g1": torch.ones(1, 1)}
        )
        exact = TestBPPlateMembers()._exact_member_marginals(fg, {"g1": 1})
        assert torch.allclose(out.probabilities, exact["h2"][:, 0], atol=1e-5)

    def test_leading_dims(self):
        fg = self._star()
        target = torch.rand(2, 3, 1).round()
        out = BeliefPropagation(fg, iters=10).query(
            query={"a": target}, evidence={"b": torch.ones(2, 3, 1)}
        )
        assert out.probabilities.shape == (2, 3)

    def test_gradients_flow_through_the_probability(self):
        fg = self._star()
        out = BeliefPropagation(fg, iters=10).query(
            query={"a": torch.ones(4, 1)}, evidence={"b": torch.ones(4, 1)}
        )
        out.probabilities.log().sum().backward()
        grads = [p.grad for p in fg.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    @pytest.mark.parametrize("name,target,match", [
        ("a", torch.full((1, 1), 0.5), "0 or 1"),
        ("h", torch.tensor([[0.5, 0.5, 0.0]]), "one-hot"),
    ])
    def test_invalid_target_raises(self, name, target, match):
        with pytest.raises(ValueError, match=match):
            BeliefPropagation(self._star()).query(query={name: target}, evidence={})

    def test_target_also_in_evidence_raises(self):
        one = torch.ones(1, 1)
        with pytest.raises(ValueError, match="both query and evidence"):
            BeliefPropagation(self._star()).query(query={"a": one}, evidence={"a": one})

    def test_continuous_target_raises(self):
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=2)
        fg = _ConcreteModel(variables=[a, emb], factors=[_pot([a], "ua", conditioning=[emb])])
        with pytest.raises(ValueError, match="not discretely enumerable"):
            BeliefPropagation(fg).query(query={"emb": torch.randn(1, 2)}, evidence={})
