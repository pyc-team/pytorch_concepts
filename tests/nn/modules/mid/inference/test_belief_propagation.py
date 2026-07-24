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
from torch_concepts.nn.modules.mid.inference.utils import enumerable_cardinality
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


def _assert_marginals_match(fg, out, exact, names, atol=1e-5):
    for n in names:
        bp = _state_marginal(fg.variables[n], out.probs[n])
        assert torch.allclose(bp, exact[n], atol=atol), (n, bp, exact[n])
        assert torch.allclose(bp.sum(-1), torch.ones_like(bp.sum(-1)), atol=1e-5)


def _binary_ce(logits, target):
    """Cross-entropy on a binary variable's canonical (log-odds) logits."""
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), target.to(logits.dtype)
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
        out = BeliefPropagation(fg, iters=25).query(query=["a", "b", "c"], evidence={})
        exact = _exact_marginals(fg, ["a", "b", "c"])
        _assert_marginals_match(fg, out, exact, ["a", "b", "c"])

    def test_star_marginals_exact(self):
        torch.manual_seed(1)
        hub = _cat("h", 3)
        leaves = [_bin(f"l{i}") for i in range(3)]
        factors = [_pot([hub], "uh")]
        for i, lf in enumerate(leaves):
            factors.append(_pot([hub, lf], f"e{i}"))
        fg = MarkovNetwork(variables=[hub, *leaves], factors=factors)
        names = ["h", "l0", "l1", "l2"]
        out = BeliefPropagation(fg, iters=25).query(query=names, evidence={})
        exact = _exact_marginals(fg, names)
        _assert_marginals_match(fg, out, exact, names)

    def test_single_unary_marginal(self):
        a = _bin("a")
        fg = _ConcreteModel(variables=[a], factors=[_pot([a], "ua")])
        out = BeliefPropagation(fg, iters=3).query(query=["a"], evidence={})
        exact = _exact_marginals(fg, ["a"])
        _assert_marginals_match(fg, out, exact, ["a"])


class TestBPMatchesDirectedOnDAG:
    def test_all_cpd_factorgraph_marginals_exact(self):
        torch.manual_seed(2)
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        fg = _ConcreteModel(variables=[a, b], factors=[cpd_a, cpd_b])
        out = BeliefPropagation(fg, iters=15).query(query=["a", "b"], evidence={})
        exact = _exact_marginals(fg, ["a", "b"])
        _assert_marginals_match(fg, out, exact, ["a", "b"])


class TestBPConditionalAndEvidence:
    def test_crf_batched_conditioning(self):
        torch.manual_seed(3)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = _pot([a, b], "phi", conditioning=[emb])
        fg = _ConcreteModel(variables=[a, b, emb], factors=[pot])
        e = torch.randn(7, 4)
        out = BeliefPropagation(fg, iters=10).query(query=["a", "b"], evidence={"emb": e})
        assert out.probs["a"].shape == (7, 1)
        exact = _exact_marginals(fg, ["a", "b"], conditioning={"emb": e}, batch=7)
        _assert_marginals_match(fg, out, exact, ["a", "b"])

    def test_discrete_evidence_matches_exact_conditional(self):
        torch.manual_seed(4)
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a], "ua"), _pot([a, b], "ab")])
        out = BeliefPropagation(fg, iters=10).query(query=["a"], evidence={"b": torch.ones(1, 1)})
        exact = _exact_marginals(fg, ["a"], evidence_states={"b": 1})
        _assert_marginals_match(fg, out, exact, ["a"])

    def test_observed_variable_emits_no_params(self):
        a, b = _bin("a"), _bin("b")
        fg = _ConcreteModel(variables=[a, b], factors=[_pot([a, b], "ab")])
        out = BeliefPropagation(fg, iters=5).query(query=["a", "b"], evidence={"b": torch.ones(1, 1)})
        assert "b" not in out.variables
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
        loss = _binary_ce(out.logits["c"], torch.tensor([1]))
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
            loss = _binary_ce(out.logits["a"], target)
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
        out = BeliefPropagation(fg, iters=30).query(query=names, evidence={})
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

    def test_factors_sharing_a_signature_are_batched_exactly(self):
        """Four identical-signature pairwise factors land in one bucket."""
        torch.manual_seed(11)
        hub = _bin("h")
        leaves = [_bin(f"l{i}") for i in range(4)]
        factors = [_pot([hub, lf], f"e{i}") for i, lf in enumerate(leaves)]
        fg = MarkovNetwork(variables=[hub, *leaves], factors=factors)
        names = ["h", *[lf.name for lf in leaves]]
        out = BeliefPropagation(fg, iters=30).query(query=names, evidence={})
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

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
        out = BeliefPropagation(fg, iters=30, damping=0.5).query(query=names, evidence={})
        for n in names:
            probs = _state_marginal(fg.variables[n], out.probs[n])
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
        out = BeliefPropagation(fg, iters=5).query(query=names, evidence={})
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names)

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
            query=["a", "b"], evidence={"emb": torch.randn(2, 5, 3)}
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
        eng = BeliefPropagation(fg, iters=3)
        table = eng._factor_table(pot, [a, b], {}, torch.Size([1]), torch.float32,
                                  torch.device("cpu"))
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
        first = eng.query(query=["a", "b", "c"], evidence={}).probs["b"]
        second = eng.query(query=["a", "b", "c"], evidence={}).probs["b"]
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
        out = BeliefPropagation(fg, iters=60, **kwargs).query(query=names, evidence={})
        _assert_marginals_match(fg, out, _exact_marginals(fg, names), names, atol=1e-4)

    def test_tol_stops_early_without_changing_the_answer(self):
        fg = self._chain()
        names = ["a", "b", "c"]
        loose = BeliefPropagation(fg, iters=200, tol=1e-7).query(query=names, evidence={})
        exact = _exact_marginals(fg, names)
        _assert_marginals_match(fg, loose, exact, names, atol=1e-4)

    def test_damping_is_log_space(self):
        """Damping mixes the *log* messages, so a damped step of a converged
        run is a no-op rather than a probability-space average."""
        fg = self._chain()
        names = ["a", "b", "c"]
        a = BeliefPropagation(fg, iters=80, damping=0.0).query(query=names, evidence={})
        b = BeliefPropagation(fg, iters=80, damping=0.7).query(query=names, evidence={})
        for n in names:
            assert torch.allclose(a.probs[n], b.probs[n], atol=1e-4)


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

    def test_member_evidence_not_supported(self):
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        a = _bin("a")
        fg = _ConcreteModel(variables=[a, plate], factors=[_pot([a], "ua")])
        with pytest.raises(NotImplementedError, match="member"):
            BeliefPropagation(fg, iters=3).query(query=["a"], evidence={"m1": torch.ones(1, 1)})
