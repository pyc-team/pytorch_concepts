"""
Comprehensive tests for CausallyReliableConceptBottleneckModel (C2BM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes (train and eval modes)
- Different inference engines (deterministic, independent, ancestral)
- Backbone and latent encoder integration
- Graph structure handling (chain, diamond, multi-output)
- Training: manual PyTorch loop and Lightning integration
- Gradient flow
- Filter methods for loss and metrics
- Edge cases
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn import (
    CausallyReliableConceptBottleneckModel,
)
from torch_concepts.nn.modules.mid.constructors.concept_graph import ConceptGraph
from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.independent import IndependentInference
from torch_concepts.nn.modules.mid.inference.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.high.base.learner import BaseLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            exogenous_size=32,
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

    def test_with_latent_encoder(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        )
        assert model.latent_size == 16

    def test_with_backbone(self, chain_graph, binary_chain_ann):
        backbone = DummyBackbone(out_features=8)
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=backbone,
        )
        assert model.backbone is not None

    def test_with_defaults(self, chain_graph):
        """Annotations without distributions — defaults should be used."""
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
        # Defaults should have been filled
        meta = model.concept_annotations.metadata
        assert meta['A']['distribution'] == Bernoulli

    def test_diamond_graph_init(self, diamond_graph, binary_diamond_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_diamond_ann,
            graph=diamond_graph,
        )
        assert model is not None

    def test_independent_train_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            train_inference=IndependentInference,
        )
        assert isinstance(model.train_inference, IndependentInference)
        assert isinstance(model.eval_inference, DeterministicInference)

    def test_ancestral_eval_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralSamplingInference,
        )
        assert isinstance(model.eval_inference, AncestralSamplingInference)

    def test_inference_kwargs_forwarded(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference_kwargs={'detach': True},
            train_inference_kwargs={'detach': False},
        )
        assert model.eval_inference.detach is True
        assert model.train_inference.detach is False


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
        out = self.model(query=['A', 'B', 'C'], x=self.x)
        assert out.shape == (4, 3)

    def test_query_subset(self):
        out = self.model(query=['B', 'C'], x=self.x)
        assert out.shape == (4, 2)

    def test_query_single(self):
        out = self.model(query=['A'], x=self.x)
        assert out.shape == (4, 1)

    def test_query_leaf_only(self):
        out = self.model(query=['C'], x=self.x)
        assert out.shape == (4, 1)

    def test_query_order_matters(self):
        """Output columns follow query order, not graph order."""
        out_ab = self.model(query=['A', 'B'], x=self.x)
        out_ba = self.model(query=['B', 'A'], x=self.x)
        # Columns should be swapped
        assert torch.allclose(out_ab[:, 0], out_ba[:, 1])
        assert torch.allclose(out_ab[:, 1], out_ba[:, 0])

    def test_batch_size_one(self):
        out = self.model(query=['A', 'B', 'C'], x=torch.randn(1, 8))
        assert out.shape == (1, 3)

    def test_large_batch(self):
        out = self.model(query=['A', 'B', 'C'], x=torch.randn(128, 8))
        assert out.shape == (128, 3)


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
        out = self.model(query=['A', 'B', 'C', 'D'], x=self.x)
        assert out.shape == (4, 4)

    def test_query_leaf(self):
        out = self.model(query=['D'], x=self.x)
        assert out.shape == (4, 1)


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
        out = self.model(query=['A', 'B', 'C'], x=self.x)
        # A=1, B=3, C=1 → total 5
        assert out.shape == (4, 5)

    def test_categorical_only(self):
        out = self.model(query=['B'], x=self.x)
        assert out.shape == (4, 3)


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
            train_inference=IndependentInference,
        )
        model.eval()
        assert model.inference is model.eval_inference

    def test_train_mode_uses_train_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            train_inference=IndependentInference,
        )
        model.train()
        assert model.inference is model.train_inference


# ===========================================================================
# return_logits
# ===========================================================================

class TestC2BMReturnLogits:
    """Test return_logits pass-through to inference."""

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, binary_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        self.x = torch.randn(4, 8)

    def test_return_logits_shape(self):
        out = self.model(query=['A', 'B', 'C'], x=self.x, return_logits=True)
        assert out.shape == (4, 3)

    def test_return_logits_vs_activated(self):
        """With return_logits the output is *not* squeezed through sigmoid."""
        logits = self.model(query=['A'], x=self.x, return_logits=True)
        probs = self.model(query=['A'], x=self.x, return_logits=False)
        # Logits can be outside [0, 1], probs are in [0, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


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
        out = model(query=['A', 'B', 'C'], x=x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_gradients_with_return_logits(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )
        x = torch.randn(4, 8, requires_grad=True)
        out = model(query=['A', 'B', 'C'], x=x, return_logits=True)
        out.sum().backward()
        assert x.grad is not None

    def test_detach_blocks_cross_level_gradients(self, chain_graph, binary_chain_ann):
        """With detach=True, children receive detached parent activations."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference_kwargs={'detach': True},
            train_inference_kwargs={'detach': True},
        )
        x = torch.randn(4, 8, requires_grad=True)
        out = model(query=['A', 'B', 'C'], x=x)
        out.sum().backward()
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

        out = model(query=['A', 'B', 'C'], x=x, return_logits=True)
        loss = loss_fn(out, y)
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
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            out = model(query=['A', 'B', 'C'], x=x, return_logits=True)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease after training"


# ===========================================================================
# Independent inference forward
# ===========================================================================

class TestC2BMIndependentInference:
    """Test forward pass with IndependentInference (requires GT)."""

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, binary_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            train_inference=IndependentInference,
        )
        self.x = torch.randn(4, 8)
        self.gt = torch.randint(0, 2, (4, 3)).float()

    def test_forward_train_with_gt(self):
        self.model.train()
        out = self.model(
            query=['A', 'B', 'C'], x=self.x,
            ground_truth=self.gt, concept_names=['A', 'B', 'C'],
        )
        assert out.shape == (4, 3)

    def test_forward_eval_no_gt(self):
        """Eval mode falls back to DeterministicInference — no GT needed."""
        self.model.eval()
        out = self.model(query=['A', 'B', 'C'], x=self.x)
        assert out.shape == (4, 3)

    def test_independent_return_logits(self):
        self.model.train()
        out = self.model(
            query=['A', 'B', 'C'], x=self.x,
            ground_truth=self.gt, concept_names=['A', 'B', 'C'],
            return_logits=True,
        )
        assert out.shape == (4, 3)


# ===========================================================================
# Ancestral sampling inference
# ===========================================================================

class TestC2BMAncestralInference:
    """Test forward pass with AncestralSamplingInference."""

    def test_sampling_produces_output(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralSamplingInference,
        )
        model.eval()
        x = torch.randn(4, 8)
        out = model(query=['A', 'B', 'C'], x=x)
        assert out.shape == (4, 3)

    def test_sampling_stochastic(self, chain_graph, binary_chain_ann):
        """Multiple forward passes should (in general) differ."""
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            inference=AncestralSamplingInference,
        )
        model.eval()
        x = torch.randn(8, 8)
        results = [model(query=['A'], x=x) for _ in range(10)]
        # At least one pair should differ (highly unlikely all 10 are identical)
        any_different = any(not torch.equal(results[0], r) for r in results[1:])
        assert any_different, "Ancestral sampling should produce varying outputs"


# ===========================================================================
# Filter methods
# ===========================================================================

class TestC2BMFilterMethods:

    @pytest.fixture(autouse=True)
    def _setup(self, chain_graph, binary_chain_ann):
        self.model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
        )

    def test_filter_output_for_loss(self):
        out = torch.randn(4, 3)
        target = torch.randint(0, 2, (4, 3)).float()
        filtered = self.model.filter_output_for_loss(out, target)
        assert 'input' in filtered
        assert 'target' in filtered
        assert torch.equal(filtered['input'], out)
        assert torch.equal(filtered['target'], target)

    def test_filter_output_for_metrics(self):
        out = torch.randn(4, 3)
        target = torch.randint(0, 2, (4, 3)).float()
        filtered = self.model.filter_output_for_metrics(out, target)
        assert 'preds' in filtered
        assert 'target' in filtered
        assert torch.equal(filtered['preds'], out)
        assert torch.equal(filtered['target'], target)


# ===========================================================================
# Lightning integration
# ===========================================================================

class TestC2BMLightning:
    """Test Lightning training capabilities."""

    def test_lightning_mode_creates_learner(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            lightning=True,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01},
        )
        assert isinstance(model, BaseLearner)

    def test_configure_optimizers(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            lightning=True,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01},
        )
        config = model.configure_optimizers()
        assert 'optimizer' in config

    def test_training_step(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            lightning=True,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01},
        )
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()},
        }
        model.train()
        loss = model.training_step(batch)
        assert loss is not None
        assert loss.requires_grad

    def test_lightning_with_independent_inference(self, chain_graph, binary_chain_ann):
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            train_inference=IndependentInference,
            lightning=True,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01},
        )
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()},
        }
        model.train()
        loss = model.training_step(batch)
        assert loss is not None
        assert loss.requires_grad


# ===========================================================================
# Backbone integration
# ===========================================================================

class TestC2BMBackbone:
    """Test C2BM with backbone feature extractor."""

    def test_backbone_forward(self, chain_graph, binary_chain_ann):
        backbone = DummyBackbone(out_features=8)
        model = CausallyReliableConceptBottleneckModel(
            input_size=8,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=backbone,
        )
        # Raw input larger than input_size, backbone projects down
        x = torch.randn(4, 32)
        out = model(query=['A', 'B', 'C'], x=x)
        assert out.shape == (4, 3)

    def test_backbone_plus_latent_encoder(self, chain_graph, binary_chain_ann):
        backbone = DummyBackbone(out_features=16)
        model = CausallyReliableConceptBottleneckModel(
            input_size=16,
            annotations=binary_chain_ann,
            graph=chain_graph,
            backbone=backbone,
            latent_encoder_kwargs={'hidden_size': 8, 'n_layers': 1},
        )
        x = torch.randn(4, 64)
        out = model(query=['A', 'B', 'C'], x=x)
        assert out.shape == (4, 3)


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
        )
        r = repr(model)
        assert 'DummyBackbone' in r
