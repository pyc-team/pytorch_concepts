"""ProbabilisticModel as the general factor graph (directed/undirected/mixed),
MarkovNetwork for undirected models, and the transparent reparenting of
BayesianNetwork (FACTOR_GRAPH_INSTRUCTIONS.md §5.5-5.6)."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.potential import ParametricPotential
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.models.markov_network import MarkovNetwork
from torch_concepts.nn.modules.low.priors import LearnablePrior


def _bin(name):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=1)


class TestProbabilisticModelGraph:
    def test_undirected_construction_and_factor_keys(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="phi_ab")
        pm = ProbabilisticModel(variables=[a, b], factors=[pot])
        assert set(pm.factors.keys()) == {"phi_ab"}
        assert isinstance(pm.factors, nn.ModuleDict)

    def test_default_potential_name(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4))
        pm = ProbabilisticModel(variables=[a, b], factors=[pot])
        assert "phi(a,b)" in pm.factors

    def test_adjacency_and_neighbors(self):
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        p_ab = ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="ab")
        p_bc = ParametricPotential(scope=[b, c], parametrization=LearnablePrior(4), name="bc")
        pm = ProbabilisticModel(variables=[a, b, c], factors=[p_ab, p_bc])
        assert {f.name for f in pm.factors_of("b")} == {"ab", "bc"}
        assert {f.name for f in pm.factors_of("a")} == {"ab"}
        assert {v.name for v in pm.neighbors("b")} == {"a", "c"}
        assert {v.name for v in pm.neighbors("a")} == {"b"}

    def test_empty_factors_allowed(self):
        a = _bin("a")
        pm = ProbabilisticModel(variables=[a])
        assert len(pm.factors) == 0

    def test_duplicate_factor_name_raises(self):
        a, b = _bin("a"), _bin("b")
        p1 = ParametricPotential(scope=[a], parametrization=LearnablePrior(2), name="dup")
        p2 = ParametricPotential(scope=[b], parametrization=LearnablePrior(2), name="dup")
        with pytest.raises(ValueError, match="duplicate factor name"):
            ProbabilisticModel(variables=[a, b], factors=[p1, p2])

    def test_unregistered_scope_variable_raises(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="phi")
        with pytest.raises(ValueError, match="not in variables list"):
            ProbabilisticModel(variables=[a], factors=[pot])

    def test_non_factor_raises(self):
        a = _bin("a")
        with pytest.raises(TypeError, match="ParametricFactor"):
            ProbabilisticModel(variables=[a], factors=[nn.Linear(2, 2)])

    def test_duplicate_variable_names_raise(self):
        a1, a2 = _bin("a"), _bin("a")
        with pytest.raises(ValueError, match="Duplicate variable names"):
            ProbabilisticModel(variables=[a1, a2], factors=[])


class TestGraphKind:
    def _cpd_pair(self):
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        return a, b, cpd_a, cpd_b

    def test_is_directed(self):
        a, b, cpd_a, cpd_b = self._cpd_pair()
        pm = ProbabilisticModel(variables=[a, b], factors=[cpd_a, cpd_b])
        assert pm.is_directed and not pm.is_undirected and not pm.is_mixed

    def test_is_undirected(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="phi")
        pm = ProbabilisticModel(variables=[a, b], factors=[pot])
        assert pm.is_undirected and not pm.is_directed and not pm.is_mixed

    def test_is_mixed(self):
        a, b, cpd_a, cpd_b = self._cpd_pair()
        c = _bin("c")
        cpd_c = ParametricCPD(variable=c, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        pot = ParametricPotential(scope=[b, c], parametrization=LearnablePrior(4), name="phi_bc")
        pm = ProbabilisticModel(variables=[a, b, c], factors=[cpd_a, cpd_b, cpd_c, pot])
        assert pm.is_mixed


class TestMarkovNetwork:
    def test_construction(self):
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        factors = [
            ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="ab"),
            ParametricPotential(scope=[b, c], parametrization=LearnablePrior(4), name="bc"),
        ]
        mn = MarkovNetwork(variables=[a, b, c], factors=factors)
        assert isinstance(mn, ProbabilisticModel)
        assert mn.is_undirected
        assert set(mn.factors.keys()) == {"ab", "bc"}

    def test_variable_can_appear_in_many_potentials(self):
        a, b, c = _bin("a"), _bin("b"), _bin("c")
        factors = [
            ParametricPotential(scope=[a, b], parametrization=LearnablePrior(4), name="ab"),
            ParametricPotential(scope=[b, c], parametrization=LearnablePrior(4), name="bc"),
            ParametricPotential(scope=[b], parametrization=LearnablePrior(2), name="ub"),
        ]
        mn = MarkovNetwork(variables=[a, b, c], factors=factors)
        assert {f.name for f in mn.factors_of("b")} == {"ab", "bc", "ub"}

    def test_rejects_cpd(self):
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        pot = ParametricPotential(scope=[b], parametrization=LearnablePrior(2), name="ub")
        with pytest.raises(TypeError, match="ParametricPotential"):
            MarkovNetwork(variables=[a, b], factors=[cpd_a, pot])


class TestBayesianNetworkReparenting:
    def _bn(self):
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        return BayesianNetwork(variables=[a, b], factors=[cpd_a, cpd_b])

    def test_bn_is_probabilistic_model(self):
        bn = self._bn()
        assert isinstance(bn, ProbabilisticModel)

    def test_bn_keeps_child_keyed_factors(self):
        bn = self._bn()
        assert set(bn.factors.keys()) == {"a", "b"}

    def test_bn_still_has_levels_and_sorted(self):
        bn = self._bn()
        assert [v.name for v in bn.sorted_variables][0] == "a"
        assert len(bn.levels) == 2

    def test_bn_rejects_potential_factor(self):
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        pot = ParametricPotential(scope=[b], parametrization=LearnablePrior(2), name="b")
        with pytest.raises(TypeError, match="ParametricCPD"):
            BayesianNetwork(variables=[a, b], factors=[cpd_a, pot])

    def test_bn_wrong_count_message_preserved(self):
        a = _bin("a")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        with pytest.raises(ValueError, match="exactly one factor per variable"):
            BayesianNetwork(variables=[a], factors=[])

    def test_bn_cycle_message_preserved(self):
        a, b = _bin("a"), _bin("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": nn.Linear(1, 1)}, parents=[b])
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": nn.Linear(1, 1)}, parents=[a])
        with pytest.raises(ValueError, match="cycle"):
            BayesianNetwork(variables=[a, b], factors=[cpd_a, cpd_b])
