"""Every engine must accept any number of leading (batch-like) dimensions.

Tensors are ``(*leading, *event)``; the event is always the last axis. The
reference for a run with several leading dimensions is the *same* run with those
dimensions flattened into one batch axis: the engines must agree element for
element, and hand back results carrying the original leading shape.
"""
import warnings

import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.distributions.delta import Delta
from torch_concepts.nn import ImportanceSampling, LearnablePrior, RejectionSampling
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.graph.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.torch.ancestral import (
    AncestralSamplingInference,
)
from torch_concepts.nn.modules.mid.inference.torch.belief_propagation import (
    BeliefPropagation,
)
from torch_concepts.nn.modules.mid.inference.torch.deterministic import (
    DeterministicInference,
)
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.mutilated_network import (
    MutilatedNetworkProposal,
)
from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable


# One, two and three leading dimensions, all holding the same 6 observations, so
# every case can be compared against the flattened (6,) run.
LEADINGS = [(6,), (2, 3), (2, 3, 1)]


@pytest.fixture
def net():
    """``x -> g`` (a plate of two binary members) ``-> y`` (binary) and ``n`` (Normal)."""
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    n = ConceptVariable("n", distribution=dist.Normal, size=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BayesianNetwork(
            variables=[x, g, y, n],
            factors=[
                ParametricCPD(variable=x, parametrization={"value": nn.Identity()}),
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


@pytest.fixture
def mrf():
    torch.manual_seed(0)
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


@pytest.fixture
def chain():
    """Two binary variables, small enough to enumerate and to sample from."""
    torch.manual_seed(1)
    c1 = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
    c2 = ConceptVariable("c2", distribution=dist.Bernoulli, size=1)
    return BayesianNetwork(
        variables=[c1, c2],
        factors=[
            ParametricCPD(variable=c1, parametrization={"logits": LearnablePrior(1)}),
            ParametricCPD(variable=c2, parametrization={"logits": nn.Linear(1, 1)}, parents=[c1]),
        ],
    )


def _engine(cls, pgm, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(pgm, **kwargs)


class TestForwardLeadingDims:
    @pytest.mark.parametrize("leading", LEADINGS)
    def test_params_match_the_flattened_run(self, net, leading):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        x = torch.randn(*leading, 4)
        out = eng.query(query=["g", "y", "n"], evidence={"x": x})
        flat = eng.query(query=["g", "y", "n"], evidence={"x": x.reshape(-1, 4)})

        assert out.logits.shape == (*leading, 3)  # m1, m2, y
        assert out.loc.shape == (*leading, 3)  # n
        assert torch.allclose(out.logits.tensor.reshape(-1, 3), flat.logits.tensor)
        assert torch.allclose(out.loc.tensor.reshape(-1, 3), flat.loc.tensor)

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_teacher_forcing_targets_carry_leading_dims(self, net, leading):
        eng = _engine(DeterministicInference, net, p_int=1.0)
        out = eng.query(
            query={"y": torch.rand(*leading, 1).round()},
            evidence={"x": torch.randn(*leading, 4)},
        )
        assert out.logits.shape == (*leading, 1)

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_samples_carry_leading_dims(self, net, leading):
        eng = _engine(AncestralSamplingInference, net, p_int=0.0)
        out = eng.query(query=["y"], evidence={"x": torch.randn(*leading, 4)})
        # Samples cover every variable drawn on the way to y, not just the query.
        assert out.samples.shape == (*leading, 3)
        assert "g" in out.samples and "m1" in out.samples

    def test_gradients_flow_with_several_leading_dims(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        out = eng.query(query=["y"], evidence={"x": torch.randn(2, 3, 4)})
        out.logits.tensor.sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_ancestral_backward_in_training_mode(self, net, leading):
        # Regression + coverage: in training mode the engine advances its
        # relaxation-temperature buffer, which several relaxed samples (the plate
        # ``g`` and ``y``) share within one query. The loss below flows back
        # through those samples into that buffer, so a backward must succeed for
        # any number of leading dims. (An in-place temperature update used to
        # corrupt this graph and raise an in-place-modification error.)
        eng = _engine(AncestralSamplingInference, net, p_int=0.0)
        eng.train()
        out = eng.query(query=["g", "y", "n"], evidence={"x": torch.randn(*leading, 4)})
        loss = out.logits.tensor.pow(2).mean() + out.loc.tensor.pow(2).mean()
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in net.parameters()
        )

    def test_mismatched_leading_shapes_raise(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        with pytest.raises(ValueError, match="mismatched leading"):
            eng.query(
                query=["y"],
                evidence={"x": torch.randn(2, 3, 4), "g": torch.rand(2, 4, 2)},
            )


class TestBeliefPropagationLeadingDims:
    @pytest.mark.parametrize("leading", LEADINGS)
    def test_undirected_marginals_match_the_flattened_run(self, mrf, leading):
        eng = BeliefPropagation(mrf, iters=8)
        a = torch.rand(*leading, 1).round()
        out = eng.query(query=["b"], evidence={"a": a})
        flat = eng.query(query=["b"], evidence={"a": a.reshape(-1, 1)})
        assert out.probs.shape == (*leading, 1)
        assert torch.allclose(out.probs.tensor.reshape(-1, 1), flat.probs.tensor, atol=1e-5)

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_directed_marginals_match_the_flattened_run(self, chain, leading):
        eng = BeliefPropagation(chain, iters=10)
        ev = torch.rand(*leading, 1).round()
        out = eng.query(query=["c1"], evidence={"c2": ev})
        flat = eng.query(query=["c1"], evidence={"c2": ev.reshape(-1, 1)})
        assert out.probs.shape == (*leading, 1)
        assert torch.allclose(out.probs.tensor.reshape(-1, 1), flat.probs.tensor, atol=1e-5)

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_gradients_flow_with_several_leading_dims(self, chain, leading):
        # The whole message-passing pass is differentiable; a marginal loss must
        # reach the factor parametrizations for any number of leading dims.
        eng = BeliefPropagation(chain, iters=5)
        out = eng.query(query=["c1"], evidence={"c2": torch.rand(*leading, 1).round()})
        out.probs.tensor.pow(2).mean().backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in chain.parameters()
        )


class TestEstimatorLeadingDims:
    """The estimators collapse the leading dims internally; the estimate must
    still come back shaped like them."""

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_rejection_sampling(self, chain, leading):
        eng = _engine(RejectionSampling, chain, n_samples=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = eng.query(query={"c2": torch.rand(*leading, 1).round()}, evidence={})
        assert out.probabilities.shape == leading

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_importance_sampling(self, chain, leading):
        eng = _engine(
            ImportanceSampling, chain,
            proposal=MutilatedNetworkProposal(chain), n_samples=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = eng.query(query={"c2": torch.rand(*leading, 1).round()}, evidence={})
        assert out.probabilities.shape == leading


class TestAnnotatedOutput:
    """The quantity-keyed output: labels, types, and views rather than copies."""

    def test_labels_and_quantities(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        out = eng.query(query=["g", "y", "n"], evidence={"x": torch.randn(2, 3, 4)})
        assert set(out.quantities) == {"logits", "loc", "scale"}
        assert out.logits.annotation.labels == ["m1", "m2", "y"]
        assert out.loc.annotation.labels == ["n"]

    def test_label_slice_is_a_view_not_a_copy(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        out = eng.query(query=["g", "y"], evidence={"x": torch.randn(2, 3, 4)})
        whole = out.logits.tensor.untyped_storage().data_ptr()
        assert out.logits["m1"].tensor.untyped_storage().data_ptr() == whole
        assert out.logits["y"].tensor.untyped_storage().data_ptr() == whole

    def test_plate_name_addresses_its_members(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        out = eng.query(query=["g", "y"], evidence={"x": torch.randn(2, 3, 4)})
        # 'g' is not a label — its members are — but it still slices the block.
        assert "g" not in out.logits.annotation.labels
        assert out.logits["g"].shape == (2, 3, 2)
        assert torch.equal(out.logits["g"].tensor, out.logits["m1", "m2"].tensor)

    def test_split_by_type(self, net):
        eng = _engine(DeterministicInference, net, p_int=0.0)
        out = eng.query(query=["g", "y", "n"], evidence={"x": torch.randn(5, 4)})
        assert out.logits.binary().annotation.labels == ["m1", "m2", "y"]
        assert out.loc.continuous().annotation.labels == ["n"]


class TestAnnotationSurvivesAcrossEngines:
    """Labels must survive the leading-dim round trip for *every* engine that
    returns annotated tensors, not only the deterministic forward pass."""

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_ancestral_labels_survive(self, net, leading):
        eng = _engine(AncestralSamplingInference, net, p_int=0.0)
        out = eng.query(query=["g", "y", "n"], evidence={"x": torch.randn(*leading, 4)})
        assert out.logits.annotation.labels == ["m1", "m2", "y"]
        assert out.samples.annotation.labels == ["m1", "m2", "y", "n"]
        assert out.samples.shape == (*leading, 6)  # m1,m2,y (1 each) + n (3)

    @pytest.mark.parametrize("leading", LEADINGS)
    def test_belief_propagation_labels_survive(self, chain, leading):
        eng = BeliefPropagation(chain, iters=5)
        out = eng.query(query=["c1"], evidence={"c2": torch.rand(*leading, 1).round()})
        assert out.probs.annotation.labels == ["c1"]
        assert out.probs["c1"].shape == (*leading, 1)


class TestPyroVariationalLeadingDims:
    """The Pyro backend collapses the leading dims into one batch axis and
    restores them on the reported tensors. Runs only where pyro is installed."""

    @staticmethod
    def _engine_and_pgm():
        torch.manual_seed(0)
        x = EmbeddingVariable("x", distribution=Delta, size=3)
        z = ConceptVariable("z", distribution=dist.Bernoulli, size=1)
        y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pgm = BayesianNetwork(
                variables=[x, y, z],
                factors=[
                    # Pyro traces the root (it does not clamp-and-skip observed
                    # roots the way the forward engines do), so the root CPD is
                    # actually called with no args — it needs a real prior.
                    ParametricCPD(variable=x, parametrization=LearnablePrior(3)),
                    ParametricCPD(variable=z, parametrization={"logits": nn.Linear(3, 1)}, parents=[x]),
                    ParametricCPD(variable=y, parametrization={"logits": nn.Linear(1, 1)}, parents=[z]),
                ],
            )
            guide = ParametricCPD(variable=z, parametrization={"logits": nn.Linear(3, 1)}, parents=[x])
            from torch_concepts.nn import VariationalInference
            eng = VariationalInference(pgm, latents={"z": guide})
        return eng, pgm

    @pytest.mark.parametrize("leading", [(4,), (2, 3), (2, 3, 1)])
    def test_restore_shapes_and_labels(self, leading):
        pytest.importorskip("pyro")
        eng, _ = self._engine_and_pgm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = eng.query(query={"y": torch.zeros(*leading, 1), "z": None},
                            evidence={"x": torch.randn(*leading, 3)})
        for tensor in list(out.params.values()) + list(out.guide_params.values()):
            assert tensor.shape[:len(leading)] == leading
            assert hasattr(tensor, "annotation")

    def test_matches_the_flattened_run(self):
        pytest.importorskip("pyro")
        import pyro
        eng, _ = self._engine_and_pgm()
        x = torch.randn(2, 3, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyro.set_rng_seed(0); torch.manual_seed(0)
            multi = eng.query(query={"y": torch.zeros(2, 3, 1), "z": None}, evidence={"x": x})
            pyro.set_rng_seed(0); torch.manual_seed(0)
            flat = eng.query(query={"y": torch.zeros(6, 1), "z": None}, evidence={"x": x.reshape(6, 3)})
        for key in multi.params:
            assert torch.allclose(
                multi.params[key].tensor.reshape(6, -1), flat.params[key].tensor, atol=1e-5
            )

    @pytest.mark.parametrize("leading", [(4,), (2, 3)])
    def test_backward_with_leading_dims(self, leading):
        pytest.importorskip("pyro")
        eng, pgm = self._engine_and_pgm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = eng.query(query={"y": torch.zeros(*leading, 1), "z": None},
                            evidence={"x": torch.randn(*leading, 3)})
        loss = sum(t.tensor.pow(2).mean() for t in out.params.values())
        loss = loss + sum(t.tensor.pow(2).mean() for t in out.guide_params.values())
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in pgm.parameters()
        )
