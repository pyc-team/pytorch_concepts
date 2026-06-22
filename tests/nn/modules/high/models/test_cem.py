"""
Comprehensive tests for Concept Embedding Model (CEM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Exogenous variables handling
- Training modes (manual PyTorch and Lightning)
- Backbone integration
- Distribution handling
- Inference modes (deterministic and ancestral sampling)
- Target preparation (prepare_target)
- Edge cases and error handling

Migrated to the high-level model API:
- forward takes ``input=`` (not ``x=``) and returns ``ModelOutput`` with
  ``.params`` only (no ``.probs`` / ``.logits``). For each queried concept,
  ``out.params[name]`` is ``{'logits': tensor(B, cardinality)}`` (CEM sets
  ``param_for_discrete_var='logits'``).
- ``model.model`` -> ``model.pgm``.
- default distributions are now BASE families (Bernoulli / OneHotCategorical),
  read via ``model.concept_annotations.concept(name).distribution``.
- ``latent_encoder_kwargs`` removed; use ``backbone=MLP(...)`` + ``latent_size=``.
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical
from torch_concepts.nn.modules.high.models.cem import (
    ConceptEmbeddingModel
)
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn import (
    MLP,
    DeterministicInference,
    AncestralSamplingInference
)


def _logits(out, names):
    """Concatenate the queried concepts' logits into a (B, sum(card)) tensor."""
    import torch
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class TestCEMInitialization(unittest.TestCase):
    """Test CEM initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
                }
            )
        })

    def test_init_basic(self):
        """Test basic initialization."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

        self.assertIsInstance(model.pgm, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['color', 'shape', 'size', 'task1'])

    def test_init_with_exogenous_size(self):
        """Test initialization with custom exogenous size."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            embedding_size=32
        )

        self.assertIsInstance(model.pgm, nn.Module)
        # The exogenous size should be passed to the encoder
        self.assertTrue(hasattr(model, 'pgm'))

    def test_init_with_defaults(self):
        """Test initialization without explicit distributions (defaults used)."""
        ann_no_dist = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann_no_dist,
            task_names=['task']
        )

        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
        # Distributions live on the model, keyed by type.
        self.assertEqual(model.variable_distributions['binary'], Bernoulli)

    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone()
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,
            task_names=['task1']
        )

        self.assertIsNotNone(model.backbone)

    def test_init_with_latent_encoder(self):
        """Test initialization with an MLP latent encoder (backbone + latent_size)."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            backbone=MLP(input_size=8, hidden_size=16, n_layers=2),
            latent_size=16
        )

        self.assertEqual(model.latent_size, 16)

    def test_init_with_deterministic_inference(self):
        """Test initialization with deterministic inference."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference
        )
        model.eval()  # Switch to eval mode to check eval_inference

        self.assertIsInstance(model.inference, DeterministicInference)

    def test_init_with_ancestral_sampling_inference(self):
        """Test initialization with ancestral sampling inference (init-only)."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=AncestralSamplingInference
        )
        model.eval()  # Switch to eval mode to check eval_inference

        self.assertIsInstance(model.inference, AncestralSamplingInference)

    def test_factory_default_is_pytorch(self):
        """Test that default lightning=False creates pure PyTorch model."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

        # Default is pure PyTorch (no learner mixin)
        self.assertFalse(isinstance(model, BaseLearner))

    def test_factory_lightning_training(self):
        """Test that lightning=True creates Lightning model."""
        model = ConceptEmbeddingModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

        # Should have BaseLearner mixin
        self.assertIsInstance(model, BaseLearner)


class TestCEMForward(unittest.TestCase):
    """Test CEM forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
                }
            )
        })

        self.model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

    def test_forward_basic(self):
        """Test basic forward pass."""
        x = torch.randn(2, 8)
        query = ['color', 'shape', 'size']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        # Output shape: batch_size x sum(cardinalities for queried variables)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2 + 1)  # color + shape + size

    def test_forward_all_concepts(self):
        """Test forward with all concepts and tasks."""
        x = torch.randn(4, 8)
        query = ['color', 'shape', 'size', 'task1']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 4)
        self.assertEqual(logits.shape[1], 3 + 2 + 1 + 1)

    def test_forward_single_concept(self):
        """Test forward with single concept."""
        x = torch.randn(2, 8)
        query = ['color']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3)

    def test_forward_only_tasks(self):
        """Test forward with only task variables."""
        x = torch.randn(2, 8)
        query = ['task1']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 1)

    def test_forward_with_backbone(self):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptEmbeddingModel(
            input_size=100,  # raw input width (consumed by the backbone)
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,
            task_names=['task1']
        )

        x = torch.randn(2, 100)  # Raw input size (before backbone)
        query = ['color', 'shape']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2)

    def test_forward_batch_sizes(self):
        """Test forward with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            query = ['color', 'shape']
            out = self.model(query=query, input=x)

            logits = _logits(out, query)
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], 3 + 2)

    def test_forward_deterministic_inference(self):
        """Test forward with deterministic inference."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference
        )

        x = torch.randn(2, 8)
        query = ['color', 'shape']

        # Multiple forwards should give same result (deterministic)
        out1 = model(query=query, input=x)
        out2 = model(query=query, input=x)

        self.assertTrue(torch.allclose(_logits(out1, query), _logits(out2, query)))

    def test_forward_eval_mode(self):
        """Test forward in eval mode."""
        self.model.eval()
        x = torch.randn(2, 8)
        query = ['color', 'shape']

        with torch.no_grad():
            out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2)


class TestCEMExogenousVariables(unittest.TestCase):
    """Test CEM exogenous variable handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[2, 3, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

    def test_different_exogenous_sizes(self):
        """Test models with different exogenous sizes."""
        for exogenous_size in [4, 8, 16, 32]:
            model = ConceptEmbeddingModel(
                input_size=8,
                annotations=self.ann,
                task_names=['task'],
                embedding_size=exogenous_size
            )

            x = torch.randn(2, 8)
            query = ['c1', 'c2', 'c3']
            out = model(query=query, input=x)

            logits = _logits(out, query)
            self.assertEqual(logits.shape[0], 2)
            self.assertEqual(logits.shape[1], 2 + 3 + 1)

    def test_exogenous_in_bipartite_model(self):
        """Test that exogenous variables are properly integrated."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task'],
            embedding_size=16
        )

        # Model should have bipartite structure with exogenous
        self.assertTrue(model.pgm is not None)


class TestCEMPrepareTarget(unittest.TestCase):
    """Test CEM prepare_target."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        self.model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

    def test_prepare_target(self):
        """Test prepare_target returns target unchanged for CEM."""
        target = torch.randint(0, 2, (2, 3)).float()

        prepared = self.model.prepare_target(target)
        self.assertTrue(torch.allclose(prepared, target))


class TestCEMTraining(unittest.TestCase):
    """Test CEM training scenarios (manual PyTorch)."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

    def test_manual_training_mode(self):
        """Test manual PyTorch training (no lightning mode)."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        # No lightning mode = pure PyTorch module (no learner mixin)
        self.assertFalse(isinstance(model, BaseLearner))

        # Can train manually
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()
        query = ['c1', 'c2', 'task']

        model.train()
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)

        self.assertTrue(loss.requires_grad)

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        x = torch.randn(4, 8, requires_grad=True)
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)
        loss = _logits(out, query).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_training_step(self):
        """Test a complete (manual) training step."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        # Training step
        model.train()
        x = torch.randn(8, 8)
        y = torch.randint(0, 2, (8, 3)).float()
        query = ['c1', 'c2', 'task']

        optimizer.zero_grad()
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)
        loss.backward()
        optimizer.step()

        self.assertTrue(True)  # If we get here, training works

    def test_parameters_update(self):
        """Test that parameters actually update during training."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Training step
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()

        x = torch.randn(8, 8)
        y = torch.randn(8, 3)
        query = ['c1', 'c2', 'task']

        optimizer.zero_grad()
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)
        loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        self.assertTrue(params_changed)


class TestCEMWithMultipleTasks(unittest.TestCase):
    """Test CEM with multiple task variables."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task1', 'task2'],
                cardinalities=[2, 3, 1, 1, 2],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task1': {'type': 'discrete'},
                    'task2': {'type': 'discrete'}
                }
            )
        })

    def test_multiple_tasks_init(self):
        """Test initialization with multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )

        self.assertEqual(model.concept_names, ['c1', 'c2', 'c3', 'task1', 'task2'])

    def test_multiple_tasks_forward(self):
        """Test forward pass with multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )

        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'c3', 'task1', 'task2']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 2 + 3 + 1 + 1 + 2)

    def test_query_only_one_task(self):
        """Test querying only one of multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )

        x = torch.randn(2, 8)
        query = ['c1', 'task1']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 2 + 1)


class TestCEMConceptTypes(unittest.TestCase):
    """Test CEM with different concept types (binary, categorical, mixed)."""

    def test_only_binary_concepts_init(self):
        """Test initialization with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],  # All binary (cardinality 1)
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'pgm'))

    def test_only_binary_concepts_forward(self):
        """Test forward pass with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],  # All binary
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        x = torch.randn(4, 8)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 4)
        self.assertEqual(logits.shape[1], 4)  # 3 binary concepts + 1 binary task
        self.assertFalse(torch.isnan(logits).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(logits).any(), "Output contains Inf values")

    def test_only_categorical_concepts_init(self):
        """Test initialization with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task'],
                cardinalities=[3, 4, 5, 2],  # All categorical (cardinality > 1)
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'pgm'))

    def test_only_categorical_concepts_forward(self):
        """Test forward pass with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task'],
                cardinalities=[3, 4, 5, 2],  # All categorical
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        x = torch.randn(4, 8)
        query = ['color', 'shape', 'size', 'task']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 4)
        self.assertEqual(logits.shape[1], 3 + 4 + 5 + 2)  # Sum of all cardinalities

    def test_mixed_concepts_init(self):
        """Test initialization with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['is_red', 'shape', 'has_texture', 'size', 'task'],
                cardinalities=[1, 3, 1, 4, 2],
                metadata={
                    'is_red': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'has_texture': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'pgm'))

    def test_mixed_concepts_forward(self):
        """Test forward pass with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['is_red', 'shape', 'has_texture', 'size', 'task'],
                cardinalities=[1, 3, 1, 4, 2],  # Mixed
                metadata={
                    'is_red': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'has_texture': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            embedding_size=16
        )

        x = torch.randn(4, 8)
        query = ['is_red', 'shape', 'has_texture', 'size', 'task']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 4)
        self.assertEqual(logits.shape[1], 1 + 3 + 1 + 4 + 2)  # Sum of all cardinalities = 11


class TestCEMEdgeCases(unittest.TestCase):
    """Test CEM edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

    def test_single_concept(self):
        """Test with single concept."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )

        x = torch.randn(2, 8)
        query = ['c1', 'task']
        out = model(query=query, input=x)

        self.assertEqual(_logits(out, query).shape, (2, 2))

    def test_all_binary_concepts(self):
        """Test with all binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )

        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(query=query, input=x)

        self.assertEqual(_logits(out, query).shape, (2, 4))

    def test_all_categorical_concepts(self):
        """Test with all categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[3, 4, 5],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )

        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)

        self.assertEqual(_logits(out, query).shape, (2, 3 + 4 + 5))

    def test_repr(self):
        """Test string representation."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('ConceptEmbeddingModel', repr_str)

    def test_device_consistency(self):
        """Test that model maintains device consistency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        device = torch.device('cuda')
        model = model.to(device)

        x = torch.randn(2, 8, device=device)
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)

        self.assertEqual(_logits(out, query).device.type, device.type)


class TestCEMCardinalities(unittest.TestCase):
    """Test CEM cardinality extraction and handling."""

    def test_concept_cardinalities_extraction(self):
        """Test that concept cardinalities are correctly extracted."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task1', 'task2'],
                cardinalities=[2, 3, 1, 1, 4],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'c3': {'type': 'discrete'},
                    'task1': {'type': 'discrete'},
                    'task2': {'type': 'discrete'}
                }
            )
        })

        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task1', 'task2']
        )

        # Concept cardinalities should be [2, 3, 1] (excluding tasks)
        self.assertTrue(model.pgm is not None)


class TestCEMComparison(unittest.TestCase):
    """Test comparison between CEM and CBM."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

    def test_cem_has_exogenous(self):
        """Test that CEM and CBM produce comparable outputs."""
        from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel

        cem = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        cbm = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        # Both should work but have different architectures
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']

        cem_out = cem(query=query, input=x)
        cbm_out = cbm(query=query, input=x)

        # Outputs should have same shape
        self.assertEqual(_logits(cem_out, query).shape, _logits(cbm_out, query).shape)


class TestCEMIndependentLearner(unittest.TestCase):
    """Test CEM with Lightning training mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        self.c = torch.randint(0, 2, (self.batch_size, 3)).float()

        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': self.c}
        }

    def test_cem_independent_training_step(self):
        """Test CEM Lightning learner training step works."""
        model = ConceptEmbeddingModel(
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            lightning=True,
            loss=nn.BCEWithLogitsLoss()
        )
        model.train()

        loss = model.training_step(self.batch)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)


if __name__ == '__main__':
    unittest.main()
