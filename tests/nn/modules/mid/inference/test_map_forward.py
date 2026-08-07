"""``MAPForwardInference``: the greedy per-node MAP forward sweep.

Covers the mode dispatcher (:func:`mode_value`) family by family, and the
engine's own contract: hard realisations in ``out.samples`` while ``out.params``
keeps the raw CPD parameters, evidence clamping, leading dimensions, and the
absence of any autograd graph.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts import seed_everything
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.low.priors import FixedPrior, LearnablePrior
from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.graph.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.torch.map_forward import MAPForwardInference
from torch_concepts.nn.modules.mid.inference.torch.utils import mode_value


LEADINGS = [(6,), (2, 3), (2, 3, 1)]


def _mixed_model():
    """x (delta root) -> g (plate: [m1, m2], bernoulli) -> y (bernoulli), n (normal)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    n = ConceptVariable("n", distribution=dist.Normal, size=3)
    return BayesianNetwork(
        variables=[x, g, y, n],
        factors=[
            ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))}),
            # logits (not probs), so ``out.params`` is visibly un-thresholded
            ParametricCPD(variable=g, parametrization={"logits": nn.Linear(4, 2)}, parents=[x]),
            ParametricCPD(variable=y, parametrization={"logits": nn.Linear(2, 1)}, parents=[g]),
            ParametricCPD(
                variable=n,
                parametrization={
                    "loc": nn.Linear(2, 3),
                    "scale": nn.Sequential(nn.Linear(2, 3), nn.Softplus()),
                },
                parents=[g],
            ),
        ],
    )


def _categorical_model():
    """x (delta root) -> k (one-hot categorical, 3 classes)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    k = ConceptVariable("k", distribution=dist.OneHotCategorical, size=3)
    return BayesianNetwork(
        variables=[x, k],
        factors=[
            ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))}),
            ParametricCPD(variable=k, parametrization={"logits": nn.Linear(4, 3)}, parents=[x]),
        ],
    )


def _categorical_plate_model():
    """x (delta root) -> p (plate: [a, b], one-hot categorical x 3 classes -> width 6)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    p = ConceptVariable("p", members=["a", "b"], distribution=dist.OneHotCategorical, size=3)
    return BayesianNetwork(
        variables=[x, p],
        factors=[
            ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))}),
            ParametricCPD(variable=p, parametrization={"logits": nn.Linear(4, 6)}, parents=[x]),
        ],
    )


def _mrf():
    """Undirected two-binary model — not a BayesianNetwork."""
    a = ConceptVariable("a", distribution=dist.Bernoulli, size=1)
    b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
    return MarkovNetwork(
        variables=[a, b],
        factors=[
            ParametricPotential(
                scope=[a, b],
                parametrization=nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)),
                name="phi_ab",
            )
        ],
    )


# ===========================================================================
# 1. Construction
# ===========================================================================
class TestConstruction:
    def test_mode_and_flags(self):
        eng = MAPForwardInference(_mixed_model())
        assert eng.name == "MAPForwardInference"
        # ``mode`` is overridden: the values are realisations, but not draws.
        assert eng.mode == "map"
        # gates the reporting of realisations in ``out.samples``
        assert eng.is_stochastic is True

    def test_repr_shows_only_its_own_knobs(self):
        r = repr(MAPForwardInference(_mixed_model()))
        assert "mode='map'" in r
        assert "parallelize_levels=False" in r
        # the inherited training knobs are pinned and inert -> not advertised
        assert "annealing" not in r
        assert "p_int" not in r

    def test_requires_bayesian_network(self):
        with pytest.raises(TypeError, match="BayesianNetwork"):
            MAPForwardInference(_mrf())

    def test_teacher_forcing_is_not_exposed(self):
        with pytest.raises(TypeError):
            MAPForwardInference(_mixed_model(), p_int=1.0)


# ===========================================================================
# 2. The values really are modes
# ===========================================================================
class TestHardValues:
    def test_binary_samples_are_hard_and_match_thresholded_logits(self):
        seed_everything(0)
        m = _mixed_model()
        eng = MAPForwardInference(m)
        out = eng.query(query=["g", "y"], evidence={"x": torch.randn(5, 4)})
        s = out.samples
        assert bool(((s == 0.0) | (s == 1.0)).all())
        # exactly the threshold of the reported (raw) logits
        expected = (torch.sigmoid(out.logits["g"]) > 0.5).to(s.dtype)
        assert torch.equal(s["g"], expected)

    def test_normal_mode_is_loc(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["n"], evidence={"x": torch.randn(4, 4)})
        assert torch.equal(out.samples["n"], out.loc["n"])

    def test_categorical_sample_is_one_hot(self):
        seed_everything(0)
        out = MAPForwardInference(_categorical_model()).query(
            query=["k"], evidence={"x": torch.randn(7, 4)}
        )
        k = out.samples["k"]
        assert k.shape == (7, 3)
        assert torch.equal(k.sum(-1), torch.ones(7))
        assert torch.equal(k.argmax(-1), out.logits["k"].argmax(-1))

    def test_categorical_plate_argmaxes_per_member_not_across_the_width(self):
        """The anti-flattening case: a single 6-way argmax would give one 1 per row."""
        seed_everything(0)
        out = MAPForwardInference(_categorical_plate_model()).query(
            query=["p"], evidence={"x": torch.randn(5, 4)}
        )
        p = out.samples["p"]
        assert p.shape == (5, 6)
        assert torch.equal(p[..., :3].sum(-1), torch.ones(5))
        assert torch.equal(p[..., 3:].sum(-1), torch.ones(5))
        assert torch.equal(p.sum(-1), 2 * torch.ones(5))  # one per member

    def test_ties_resolve_to_the_lowest_state(self):
        b = ConceptVariable("b", distribution=dist.Bernoulli, size=2)
        assert torch.equal(
            mode_value(b, {"probs": torch.tensor([[0.5, 0.5]])}), torch.zeros(1, 2)
        )
        k = ConceptVariable("k", distribution=dist.OneHotCategorical, size=3)
        assert torch.equal(
            mode_value(k, {"probs": torch.ones(1, 3) / 3}),
            torch.tensor([[1.0, 0.0, 0.0]]),
        )


# ===========================================================================
# 3. It differs from DeterministicInference (which propagates the soft param)
# ===========================================================================
class TestDiffersFromDeterministic:
    def test_soft_versus_hard(self):
        seed_everything(0)
        m = _mixed_model()
        x = torch.randn(6, 4)
        soft = DeterministicInference(m).query(query=["g"], evidence={"x": x})
        hard = MAPForwardInference(m).query(query=["g"], evidence={"x": x})

        assert soft.samples is None            # propagates a parameter, not a realisation
        assert hard.samples is not None
        probs = torch.sigmoid(soft.logits["g"])
        assert bool(((probs > 0.0) & (probs < 1.0)).all())   # genuinely soft
        assert torch.equal(hard.samples["g"], (probs > 0.5).to(probs.dtype))


# ===========================================================================
# 4. Evidence
# ===========================================================================
class TestEvidenceClamping:
    def test_clamped_variable_is_absent_from_params_and_samples(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(
            query=["y"], evidence={"x": torch.randn(3, 4), "g": torch.ones(3, 2)}
        )
        assert "g" not in out.variables
        assert "g" not in out.samples.annotation.labels

    def test_evidence_drives_the_downstream_map_value(self):
        seed_everything(0)
        m = _mixed_model()
        eng = MAPForwardInference(m)
        x = torch.randn(4, 4)
        ones = eng.query(query=["y"], evidence={"x": x, "g": torch.ones(4, 2)})
        zeros = eng.query(query=["y"], evidence={"x": x, "g": torch.zeros(4, 2)})
        # different clamped parents => different CPD params for y
        assert not torch.equal(ones.logits["y"], zeros.logits["y"])

    def test_observed_member_appears_in_samples_with_its_observed_value(self):
        seed_everything(0)
        m = _mixed_model()
        obs = torch.ones(4, 1)
        out = MAPForwardInference(m).query(
            query=["g", "y"], evidence={"x": torch.randn(4, 4), "m1": obs}
        )
        # member evidence is value forcing: the plate is still computed
        assert torch.equal(out.samples["m1"], obs)
        assert bool(((out.samples["m2"] == 0.0) | (out.samples["m2"] == 1.0)).all())


# ===========================================================================
# 5. Output contract
# ===========================================================================
class TestOutputContract:
    def test_samples_cover_every_computed_variable(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["y"], evidence={"x": torch.randn(2, 4)})
        # ``g`` is only an ancestor of the query, but it was realised -> reported
        assert set(out.samples.annotation.labels) >= {"m1", "m2", "y"}

    def test_params_keep_the_raw_cpd_output(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["g"], evidence={"x": torch.randn(8, 4)})
        logits = out.logits["g"]
        # raw logits, not thresholded or squashed into [0, 1]
        assert not bool(((logits >= 0.0) & (logits <= 1.0)).all())

    def test_member_samples_are_views_of_the_plate(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["g", "m1"], evidence={"x": torch.randn(2, 4)})
        whole = out.samples["g"]
        assert out.samples["m1"].untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()


# ===========================================================================
# 6. No autograd
# ===========================================================================
class TestNoGrad:
    def test_nothing_carries_a_graph(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["g", "y"], evidence={"x": torch.randn(3, 4)})
        assert out.samples.requires_grad is False
        assert out.samples.grad_fn is None
        for tensor in out.params.values():
            assert tensor.requires_grad is False
            assert tensor.grad_fn is None

    def test_no_gradient_reaches_the_model(self):
        seed_everything(0)
        m = _mixed_model()
        MAPForwardInference(m).query(query=["y"], evidence={"x": torch.randn(3, 4)})
        assert all(p.grad is None for p in m.parameters())

    def test_backward_is_impossible(self):
        seed_everything(0)
        m = _mixed_model()
        out = MAPForwardInference(m).query(query=["y"], evidence={"x": torch.randn(3, 4)})
        with pytest.raises(RuntimeError):
            out.samples.sum().backward()


# ===========================================================================
# 7. Leading dimensions
# ===========================================================================
class TestLeadingDims:
    @pytest.mark.parametrize("leading", LEADINGS)
    def test_shapes_and_equivalence_to_the_flat_run(self, leading):
        seed_everything(0)
        m = _mixed_model()
        eng = MAPForwardInference(m)
        flat = torch.randn(int(torch.tensor(leading).prod()), 4)

        nested = eng.query(query=["g", "y"], evidence={"x": flat.reshape(*leading, 4)})
        plain = eng.query(query=["g", "y"], evidence={"x": flat})

        assert nested.samples.shape == (*leading, 3)
        assert torch.equal(nested.samples.reshape(-1, 3), plain.samples)


# ===========================================================================
# 8. Determinism
# ===========================================================================
class TestDeterminism:
    def test_repeated_queries_are_identical(self):
        seed_everything(0)
        m = _mixed_model()
        eng = MAPForwardInference(m)
        x = torch.randn(5, 4)
        assert torch.equal(
            eng.query(query=["g", "y"], evidence={"x": x}).samples,
            eng.query(query=["g", "y"], evidence={"x": x}).samples,
        )

    def test_parallelize_levels_changes_nothing(self):
        seed_everything(0)
        m = _mixed_model()
        x = torch.randn(5, 4)
        serial = MAPForwardInference(m).query(query=["y", "n"], evidence={"x": x})
        parallel = MAPForwardInference(m, parallelize_levels=True).query(
            query=["y", "n"], evidence={"x": x}
        )
        assert torch.equal(serial.samples, parallel.samples)


# ===========================================================================
# 9. mode_value, family by family
# ===========================================================================
class TestModeValue:
    def test_bernoulli_probs_and_logits_agree(self):
        b = ConceptVariable("b", distribution=dist.Bernoulli, size=3)
        logits = torch.tensor([[-1.0, 0.5, 2.0]])
        assert torch.equal(
            mode_value(b, {"logits": logits}),
            mode_value(b, {"probs": torch.sigmoid(logits)}),
        )
        assert torch.equal(mode_value(b, {"logits": logits}), torch.tensor([[0.0, 1.0, 1.0]]))

    def test_relaxed_families_resolve_like_their_exact_twins(self):
        rb = ConceptVariable("rb", distribution=dist.RelaxedBernoulli, size=2)
        assert torch.equal(
            mode_value(rb, {"probs": torch.tensor([[0.2, 0.8]])}), torch.tensor([[0.0, 1.0]])
        )
        rk = ConceptVariable("rk", distribution=dist.RelaxedOneHotCategorical, size=3)
        assert torch.equal(
            mode_value(rk, {"probs": torch.tensor([[0.1, 0.2, 0.7]])}),
            torch.tensor([[0.0, 0.0, 1.0]]),
        )

    def test_normal_and_delta_pass_through_untouched(self):
        n = ConceptVariable("n", distribution=dist.Normal, size=2)
        loc = torch.tensor([[1.5, -2.0]])
        assert torch.equal(mode_value(n, {"loc": loc, "scale": torch.ones(1, 2)}), loc)
        d = ConceptVariable("d", distribution=Delta, size=2)
        value = torch.tensor([[7.0, 8.0]])
        assert torch.equal(mode_value(d, {"value": value}), value)

    def test_plain_categorical_is_one_hot_of_width_size(self):
        c = ConceptVariable("c", distribution=dist.Categorical, size=3)
        out = mode_value(c, {"probs": torch.tensor([[0.1, 0.2, 0.7]])})
        assert out.shape == (1, 3)
        assert torch.equal(out, torch.tensor([[0.0, 0.0, 1.0]]))

    def test_bernoulli_plate_is_elementwise(self):
        p = ConceptVariable("p", members=["m1", "m2"], distribution=dist.Bernoulli)
        assert torch.equal(
            mode_value(p, {"probs": torch.tensor([[0.9, 0.1]])}), torch.tensor([[1.0, 0.0]])
        )

    def test_categorical_plate_folds_per_member(self):
        p = ConceptVariable("p", members=["a", "b"], distribution=dist.OneHotCategorical, size=3)
        probs = torch.tensor([[0.1, 0.7, 0.2, 0.5, 0.3, 0.2]])
        assert torch.equal(
            mode_value(p, {"probs": probs}),
            torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]),
        )

    def test_dtype_is_preserved(self):
        b = ConceptVariable("b", distribution=dist.Bernoulli, size=2)
        out = mode_value(b, {"probs": torch.tensor([[0.3, 0.7]], dtype=torch.float64)})
        assert out.dtype == torch.float64


# ===========================================================================
# 10. The sibling engines are unaffected (regression guard for forward.py)
# ===========================================================================
class TestSiblingEnginesUnchanged:
    def test_samples_still_gated_by_is_stochastic(self):
        seed_everything(0)
        m = _mixed_model()
        x = torch.randn(3, 4)
        assert DeterministicInference(m).query(query=["y"], evidence={"x": x}).samples is None
        assert AncestralSamplingInference(m).query(query=["y"], evidence={"x": x}).samples is not None

    def test_modes_are_unchanged(self):
        m = _mixed_model()
        assert DeterministicInference(m).mode == "deterministic"
        assert AncestralSamplingInference(m).mode == "ancestral"
