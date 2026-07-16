"""BeliefPropagation: exactness on trees, agreement with directed inference on a
DAG, conditional (CRF) batching, evidence, differentiability, mixed graphs, and
the documented error boundaries."""
import itertools

import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.potential import ParametricPotential, enumerable_cardinality
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.torch.belief_propagation import BeliefPropagation
from torch_concepts.nn.modules.low.priors import LearnablePrior


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
    """An energy-based potential with an interaction-capable (MLP) energy."""
    return ParametricPotential(
        scope=scope,
        parametrization=_energy_net(scope, conditioning),
        conditioning=conditioning,
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
    scores = []
    for combo in itertools.product(*[range(c) for c in cards]):
        assignment = dict(ev_assign)
        for v, s in zip(free_vars, combo):
            assignment[v] = _encode(v, s, batch)
        tot = 0.0
        for f in fg.factors.values():
            tot = tot + f.log_potential(assignment, conditioning)
        scores.append(tot)
    logj = torch.stack(scores, dim=-1)  # (batch, prod)
    joint = torch.softmax(logj, dim=-1).reshape(batch, *cards)
    marg = {}
    for i, n in enumerate(free_names):
        axes = tuple(ax for ax in range(1, len(free_names) + 1) if ax != i + 1)
        marg[n] = joint.sum(dim=axes) if axes else joint
    return marg


def _assert_marginals_match(out, exact, names, atol=1e-5):
    for n in names:
        bp = out.params[n]["probs"]
        assert torch.allclose(bp, exact[n], atol=atol), (n, bp, exact[n])
        assert torch.allclose(bp.sum(-1), torch.ones_like(bp.sum(-1)), atol=1e-5)


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
        _assert_marginals_match(out, exact, ["a", "b", "c"])

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
        _assert_marginals_match(out, exact, names)

    def test_single_unary_marginal(self):
        a = _bin("a")
        fg = ProbabilisticModel(variables=[a], factors=[_pot([a], "ua")])
        out = BeliefPropagation(fg, iters=3).query(query=["a"], evidence={})
        exact = _exact_marginals(fg, ["a"])
        _assert_marginals_match(out, exact, ["a"])


class TestBPMatchesDirectedOnDAG:
    def test_all_cpd_factorgraph_marginals_exact(self):
        torch.manual_seed(2)
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        fg = ProbabilisticModel(variables=[a, b], factors=[cpd_a, cpd_b])
        out = BeliefPropagation(fg, iters=15).query(query=["a", "b"], evidence={})
        exact = _exact_marginals(fg, ["a", "b"])
        _assert_marginals_match(out, exact, ["a", "b"])


class TestBPConditionalAndEvidence:
    def test_crf_batched_conditioning(self):
        torch.manual_seed(3)
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = _pot([a, b], "phi", conditioning=[emb])
        fg = ProbabilisticModel(variables=[a, b, emb], factors=[pot])
        e = torch.randn(7, 4)
        out = BeliefPropagation(fg, iters=10).query(query=["a", "b"], evidence={"emb": e})
        assert out.params["a"]["probs"].shape == (7, 2)
        exact = _exact_marginals(fg, ["a", "b"], conditioning={"emb": e}, batch=7)
        _assert_marginals_match(out, exact, ["a", "b"])

    def test_discrete_evidence_matches_exact_conditional(self):
        torch.manual_seed(4)
        a, b = _bin("a"), _bin("b")
        fg = ProbabilisticModel(variables=[a, b], factors=[_pot([a], "ua"), _pot([a, b], "ab")])
        out = BeliefPropagation(fg, iters=10).query(query=["a"], evidence={"b": torch.ones(1, 1)})
        exact = _exact_marginals(fg, ["a"], evidence_states={"b": 1})
        _assert_marginals_match(out, exact, ["a"])

    def test_observed_variable_emits_no_params(self):
        a, b = _bin("a"), _bin("b")
        fg = ProbabilisticModel(variables=[a, b], factors=[_pot([a, b], "ab")])
        out = BeliefPropagation(fg, iters=5).query(query=["a", "b"], evidence={"b": torch.ones(1, 1)})
        assert "b" not in out.params
        assert set(out.params) == {"a"}


class TestBPTrainingAndMixed:
    def test_mixed_graph_trains(self):
        torch.manual_seed(5)
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        cpd_c = ParametricCPD(variable=c, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        pot = _pot([b, c], "phi_bc")
        fg = ProbabilisticModel(variables=[a, b, c], factors=[cpd_a, cpd_b, cpd_c, pot])
        assert fg.is_mixed
        out = BeliefPropagation(fg, iters=6).query(query=["c"], evidence={})
        loss = torch.nn.functional.cross_entropy(out.params["c"]["logits"], torch.tensor([1]))
        loss.backward()
        grads = [p.grad for p in fg.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_marginal_ce_training_reduces_loss(self):
        torch.manual_seed(6)
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=3)
        pot = _pot([a], "ua", conditioning=[emb])
        fg = ProbabilisticModel(variables=[a, emb], factors=[pot])
        eng = BeliefPropagation(fg, iters=3)
        x = torch.randn(64, 3)
        target = (x[:, 0] > 0).long()
        opt = torch.optim.Adam(fg.parameters(), lr=0.05)
        first, last = None, None
        for step in range(300):
            opt.zero_grad()
            out = eng.query(query=["a"], evidence={"emb": x})
            loss = torch.nn.functional.cross_entropy(out.params["a"]["logits"], target)
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.6


class TestBPErrors:
    def test_continuous_free_variable_raises(self):
        a = _bin("a")
        cont = ConceptVariable("z", distribution=dist.Normal, size=2)
        cpd = ParametricCPD(variable=a, parametrization={"logits": nn.Linear(2, 1)}, parents=[cont])
        cpd_z = ParametricCPD(variable=cont, parametrization={"loc": LearnablePrior(2), "scale": LearnablePrior(2)})
        fg = ProbabilisticModel(variables=[a, cont], factors=[cpd, cpd_z])
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
        fg = ProbabilisticModel(variables=[a, plate], factors=[_pot([a], "ua")])
        with pytest.raises(NotImplementedError, match="member"):
            BeliefPropagation(fg, iters=3).query(query=["a"], evidence={"m1": torch.ones(1, 1)})
