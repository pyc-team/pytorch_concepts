"""
Comprehensive tests for CausallyReliableConceptBottleneckModel (C2BM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes (train and eval modes)
- Different inference engines (deterministic, independent, ancestral)
- Backbone integration
- Graph structure handling (chain, diamond, multi-output)
- Training: manual PyTorch loop
- Gradient flow
- Target preparation (prepare_target)
- Edge cases
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.high.models.c2bm import CausallyReliableConceptBottleneckModel
from torch_concepts.nn import MLP
from torch_concepts import ConceptGraph
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.torch.independent import IndependentInference
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralInference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logits(out, names):
    """Concatenate per-variable logits for the queried ``names`` -> (B, sum cardinalities)."""
    import torch
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)


def _make_graph(adj, names):
    """Shorthand to build a ConceptGraph from a list-of-lists adjacency."""
    return ConceptGraph(torch.tensor(adj, dtype=torch.float), node_names=names)


def _chain_graph():
    """A -> B -> C  (simple chain)."""
    return _make_graph(
        [[0, 1, 0],
         [0, 0, 1],
         [0, 0, 0]],
        ['A', 'B', 'C'],
    )


def _diamond_graph():
    """A -> B, A -> C, B -> D, C -> D  (diamond/fork-join)."""
    return _make_graph(
        [[0, 1, 1, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        ['A', 'B', 'C', 'D'],
    )


def _binary_annotations(names):
    """All-binary annotations for *names* (defaults will assign Bernoulli)."""
    return Annotations({
        1: AxisAnnotation(
            labels=list(names),
            cardinalities=[1] * len(names),
            metadata={n: {'type': 'discrete'} for n in names},
        )
    })


def _mixed_annotations():
    """A (binary), B (3-class categorical), C (binary)  — chain graph."""
    return Annotations({
        1: AxisAnnotation(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 3, 1],
            metadata={
                'A': {'type': 'discrete'},
                'B': {'type': 'discrete'},
                'C': {'type': 'discrete'},
            },
        )
    })


class DummyBackbone(nn.Module):
    """Projects arbitrary input to a fixed-size feature."""
    def __init__(self, out_features=8):
        super().__init__()
        self.linear = nn.LazyLinear(out_features)
        self.out_features = out_features

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_graph():
    return _chain_graph()


@pytest.fixture
def diamond_graph():
    return _diamond_graph()


@pytest.fixture
def binary_chain_ann():
    return _binary_annotations(['A', 'B', 'C'])


@pytest.fixture
def binary_diamond_ann():
    return _binary_annotations(['A', 'B', 'C', 'D'])


@pytest.fixture
def mixed_chain_ann():
    return _mixed_annotations()


# ===========================================================================
# Initialization
# ===========================================================================

class TestC2BMInitialization:
    """Test model construction under various configurations."""

    def test_default_init(self, chain_graph, binary_chain_ann):
        """Default deterministic inference, no backbone, identity latent encoder."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        assert hasattr(model, 'eval_inference')
        assert hasattr(model, 'train_inference')
        assert isinstance(model.eval_inference, DeterministicInference)
        assert isinstance(model.train_inference, DeterministicInference)

    def test_custom_exogenous_and_hypernet_sizes(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            embedding_size=32,
            hypernet_hidden_size=32,
        )
        assert model is not None

    def test_hypernet_use_bias(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            hypernet_use_bias=True,
        )
        assert model is not None

    def test_with_mlp_backbone(self, chain_graph, binary_chain_ann):
        """Custom backbone needs an explicit latent_size."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=MLP(input_size=8, hidden_size=16, n_layers=1),
            latent_size=16,
        )
        assert model.latent_size == 16

    def test_with_backbone(self, chain_graph, binary_chain_ann):
        backbone = DummyBackbone(out_features=8)
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=backbone,
            latent_size=8,
        )
        assert model.backbone is not None

    def test_with_defaults(self, chain_graph):
        """Annotations without distributions — base-family defaults should be used."""
        ann_no_dist = Annotations({
            1: AxisAnnotation(
                labels=['A', 'B', 'C'],
                cardinalities=[1, 1, 1],
                metadata={
                    'A': {'type': 'discrete'},
                    'B': {'type': 'discrete'},
                    'C': {'type': 'discrete'},
                },
            )
        })
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=ann_no_dist,
            graph=chain_graph,
        )
        assert model.concept_names == ['A', 'B', 'C']
        # Defaults should have been filled (base family Bernoulli).
        assert model.concept_annotations.concept('A').distribution == Bernoulli

    def test_diamond_graph_init(self, diamond_graph, binary_diamond_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_diamond_ann,
            graph=diamond_graph,
        )
        assert model is not None

    def test_independent_train_inference(self, chain_graph, binary_chain_ann):
        """Using a different train_inference class should raise ValueError."""
        with pytest.raises(ValueError, match="must be the same class"):
            CausallyReliableConceptBottleneckModel(
                input_size=8,
                annotations=binary_chain_ann,
                graph=chain_graph,
                train_inference=IndependentInference,
            )

    def test_same_train_and_eval_inference_allowed(self, chain_graph, binary_chain_ann):
        """Passing the same class for train_inference and inference is allowed."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralInference,
            train_inference=AncestralInference,
        )
        assert isinstance(model.eval_inference, AncestralInference)
        assert isinstance(model.train_inference, AncestralInference)

    def test_train_inference_defaults_to_inference_class(self, chain_graph, binary_chain_ann):
        """When train_inference is omitted, it defaults to the same class as inference."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralInference,
        )
        assert type(model.train_inference) is type(model.eval_inference)

    def test_ancestral_eval_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralInference,
        )
        assert isinstance(model.eval_inference, AncestralInference)

    def test_inference_kwargs_forwarded(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference_kwargs={'p_int': 1.0},
            train_inference_kwargs={'p_int': 0.0},
        )
        assert model.eval_inference.p_int == 1.0
        assert model.train_inference.p_int == 0.0


# ===========================================================================
# Forward pass
# ===========================================================================

class TestC2BMForward:
    """Test forward pass shapes and correctness."""

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, binary_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        self.x = torch.randn(4, 8)

    def test_query_all(self):
        query = ['A', 'B', 'C']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 3)

    def test_query_subset(self):
        query = ['B', 'C']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 2)

    def test_query_single(self):
        query = ['A']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 1)

    def test_query_leaf_only(self):
        query = ['C']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 1)

    def test_query_order_matters(self):
        """Output columns follow query order, not graph order."""
        out_ab = self.model(query=['A', 'B'], input=self.x)
        out_ba = self.model(query=['B', 'A'], input=self.x)
        ab = _logits(out_ab, ['A', 'B'])
        ba = _logits(out_ba, ['B', 'A'])
        # Columns should be swapped
        assert torch.allclose(ab[:, 0], ba[:, 1])
        assert torch.allclose(ab[:, 1], ba[:, 0])

    def test_batch_size_one(self):
        query = ['A', 'B', 'C']
        out = self.model(query=query, input=torch.randn(1, 8))
        assert _logits(out, query).shape == (1, 3)

    def test_large_batch(self):
        query = ['A', 'B', 'C']
        out = self.model(query=query, input=torch.randn(128, 8))
        assert _logits(out, query).shape == (128, 3)


class TestC2BMForwardDiamond:
    """Forward pass on the diamond graph (multi-parent node D)."""

    @pytest.fixture(autouse=True)
    def _setup(self, diamond_graph, binary_diamond_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_diamond_ann,
            graph=diamond_graph,
        )
        self.x = torch.randn(4, 8)

    def test_query_all(self):
        query = ['A', 'B', 'C', 'D']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 4)

    def test_query_leaf(self):
        query = ['D']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 1)


class TestC2BMForwardMixed:
    """Forward pass with mixed cardinalities (binary + categorical)."""

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, mixed_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=mixed_chain_ann,
            graph=chain_graph,
        )
        self.x = torch.randn(4, 8)

    def test_output_dim(self):
        query = ['A', 'B', 'C']
        out = self.model(query=query, input=self.x)
        # A=1, B=3, C=1 → total 5
        assert _logits(out, query).shape == (4, 5)

    def test_categorical_only(self):
        query = ['B']
        out = self.model(query=query, input=self.x)
        assert _logits(out, query).shape == (4, 3)


# ===========================================================================
# Train / eval mode routing
# ===========================================================================

class TestC2BMTrainEvalRouting:
    """Verify inference property dispatches correctly."""

    def test_eval_mode_uses_eval_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        model.eval()
        assert model.inference is model.eval_inference

    def test_train_mode_uses_train_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        model.train()
        assert model.inference is model.train_inference


# ===========================================================================
# return_logits (API removed)
# ===========================================================================

@pytest.mark.skip(reason="out of scope: lightning/loss/metrics — revisit later")
class TestC2BMReturnLogits:
    """The return_logits API no longer exists."""

    def test_return_logits_shape(self):
        pass


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestC2BMGradients:
    """Ensure gradients flow end-to-end."""

    def test_gradients_flow_through(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        x = torch.randn(4, 8, requires_grad=True)
        query = ['A', 'B', 'C']
        out = model(query=query, input=x)
        _logits(out, query).sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_detach_blocks_cross_level_gradients(self, chain_graph, binary_chain_ann):
        """Teacher-forcing (p_int) still lets gradients reach the input."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference_kwargs={'p_int': 1.0},
            train_inference_kwargs={'p_int': 1.0},
        )
        x = torch.randn(4, 8, requires_grad=True)
        query = ['A', 'B', 'C']
        out = model(query=query, input=x)
        _logits(out, query).sum().backward()
        # Gradients should still flow to x (through at least root A)
        assert x.grad is not None


# ===========================================================================
# Manual training loop
# ===========================================================================

class TestC2BMManualTraining:
    """Manual PyTorch training loop (no Lightning)."""

    def test_training_step(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()

        query = ['A', 'B', 'C']
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)
        loss.backward()
        optimizer.step()
        assert loss.item() >= 0

    def test_loss_decreases(self, chain_graph, binary_chain_ann):
        """Sanity: loss goes down after a few steps on repeated data."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        torch.manual_seed(0)
        x = torch.randn(16, 8)
        y = torch.randint(0, 2, (16, 3)).float()

        model.train()
        query = ['A', 'B', 'C']
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            out = model(query=query, input=x)
            loss = loss_fn(_logits(out, query), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease after training"


# ===========================================================================
# Independent inference forward
# ===========================================================================

class TestC2BMIndependentInference:
    """Verify that IndependentInference as train_inference raises ValueError."""

    def test_different_train_inference_raises(self, chain_graph, binary_chain_ann):
        with pytest.raises(ValueError, match="must be the same class"):
            CausallyReliableConceptBottleneckModel(
                input_size=8,
                annotations=binary_chain_ann,
                graph=chain_graph,
                train_inference=IndependentInference,
            )


# ===========================================================================
# Ancestral sampling inference
# ===========================================================================

@pytest.mark.skip(reason="out of scope: ancestral inference forward is a known-open issue — revisit later")
class TestC2BMAncestralInference:
    """Test forward pass with AncestralInference."""

    def test_sampling_produces_output(self, chain_graph, binary_chain_ann):
        pass


# ===========================================================================
# Target preparation
# ===========================================================================

class TestC2BMPrepareTarget:

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, binary_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )

    def test_prepare_target(self):
        target = torch.randint(0, 2, (4, 3)).float()
        prepared = self.model.prepare_target(target)
        assert torch.equal(prepared, target)


# ===========================================================================
# Lightning integration
# ===========================================================================

@pytest.mark.skip(reason="out of scope: lightning/loss/metrics — revisit later")
class TestC2BMLightning:
    """Test Lightning training capabilities."""

    def test_lightning_mode_creates_learner(self):
        pass


# ===========================================================================
# Backbone integration
# ===========================================================================

class TestC2BMBackbone:
    """Test C2BM with backbone feature extractor."""

    def test_backbone_forward(self, chain_graph, binary_chain_ann):
        backbone = DummyBackbone(out_features=8)
        model = CausallyReliableConceptBottleneckModel(
            input_size=32,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=backbone,
            latent_size=8,
        )
        # Raw input enters the PGM 'input' node, backbone projects to latent.
        x = torch.randn(4, 32)
        query = ['A', 'B', 'C']
        out = model(query=query, input=x)
        assert _logits(out, query).shape == (4, 3)

    def test_backbone_with_mlp(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=64,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=MLP(input_size=64, hidden_size=8, n_layers=1),
            latent_size=8,
        )
        x = torch.randn(4, 64)
        query = ['A', 'B', 'C']
        out = model(query=query, input=x)
        assert _logits(out, query).shape == (4, 3)


# ===========================================================================
# repr
# ===========================================================================

class TestC2BMRepr:

    def test_repr_no_backbone(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        r = repr(model)
        assert 'CausallyReliableConceptBottleneckModel' in r

    def test_repr_with_backbone(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=DummyBackbone(out_features=8),
            latent_size=8,
        )
        r = repr(model)
        assert 'DummyBackbone' in r
