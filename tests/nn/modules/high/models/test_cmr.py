"""
Comprehensive tests for Concept Memory Reasoner (CMR).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- CMR-specific parameter handling
- Inference mode configuration
- Filter methods
- Factory behavior (PyTorch vs Lightning)
- Cardinality constraint checks
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts.nn.modules.high.models.cmr import ConceptMemoryReasoner
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.mid.inference import (
    DeterministicInference,
    IndependentInference,
)


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""

    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class TestCMRInitialization(unittest.TestCase):
    """Test CMR initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli},
                }
            )
        })

    def test_init_basic(self):
        """Test basic initialization."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
        )

        self.assertIsInstance(model.model, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task1'])

    def test_init_with_cmr_specific_params(self):
        """Test initialization with CMR-specific hyperparameters."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            n_rules=7,
            memory_latent_size=64,
            memory_decoder_hidden_layers=2,
            selector_hidden_layers=2,
            eps=1e-4,
        )

        self.assertIsInstance(model.model, nn.Module)
        self.assertTrue(hasattr(model, 'eval_inference'))
        self.assertTrue(hasattr(model, 'train_inference'))

    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone()
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            task_names=['task1'],
        )

        self.assertIsNotNone(model.backbone)

    def test_init_with_latent_encoder(self):
        """Test initialization with latent encoder config."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 2},
        )

        self.assertEqual(model.latent_size, 16)

    def test_init_with_deterministic_inference(self):
        """Test initialization with deterministic inference."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference,
        )
        model.eval()

        self.assertIsInstance(model.inference, DeterministicInference)

    def test_init_with_independent_train_inference(self):
        """Test initialization with independent train inference."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference,
            train_inference=IndependentInference,
        )
        model.train()

        self.assertIsInstance(model.inference, IndependentInference)

    def test_factory_default_is_pytorch(self):
        """Test that default lightning=False creates pure PyTorch model."""
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
        )

        self.assertFalse(isinstance(model, BaseLearner))

    def test_factory_lightning_training(self):
        """Test that lightning=True creates Lightning model."""
        model = ConceptMemoryReasoner(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
        )

        self.assertIsInstance(model, BaseLearner)

    def test_cardinality_constraint(self):
        """Test CMR cardinality constraint for concept variables."""
        ann_bad = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1'],
                cardinalities=[2, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli},
                }
            )
        })

        with self.assertRaises(AssertionError):
            ConceptMemoryReasoner(
                input_size=8,
                annotations=ann_bad,
                task_names=['task1'],
            )


class TestCMRForward(unittest.TestCase):
    """Test CMR forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli},
                }
            )
        })

        self.model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
        )

    def test_forward_basic(self):
        """Test basic forward pass."""
        x = torch.randn(2, 8)
        query = ['c1', 'c2']
        out = self.model(query=query, x=x)

        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 2)

    def test_forward_all_concepts(self):
        """Test forward with concepts and task."""
        x = torch.randn(4, 8)
        query = ['c1', 'c2', 'task1']
        out = self.model(query=query, x=x)

        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 3)

    def test_forward_only_task(self):
        """Test forward with only task variable."""
        x = torch.randn(3, 8)
        query = ['task1']
        out = self.model(query=query, x=x)

        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[1], 1)

    def test_forward_with_backbone(self):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptMemoryReasoner(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            task_names=['task1'],
        )

        x = torch.randn(2, 100)
        query = ['c1', 'task1']
        out = model(query=query, x=x)

        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 2)


class TestCMRFilterMethods(unittest.TestCase):
    """Test CMR filter methods."""

    def setUp(self):
        """Set up test fixtures."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli},
                }
            )
        })

        self.model = ConceptMemoryReasoner(
            input_size=8,
            annotations=ann,
            task_names=['task1'],
        )

    def test_filter_output_for_loss(self):
        """Test filter_output_for_loss returns explicit CMR loss kwargs."""
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task1']
        out_no_rec = self.model(query=query, x=x, include_rec=False, rec_weight=0.1)
        out_with_rec = self.model(query=query, x=x, include_rec=True, rec_weight=0.1)
        target = torch.randint(0, 2, out_no_rec.shape).float()

        filtered = self.model.filter_output_for_loss(
            {'no_rec': out_no_rec, 'with_rec': out_with_rec},
            target,
        )

        self.assertIsInstance(filtered, dict)
        self.assertIn('concept_input', filtered)
        self.assertIn('concept_target', filtered)
        self.assertIn('task_input', filtered)
        self.assertIn('task_input_with_rec', filtered)
        self.assertIn('task_target', filtered)

        self.assertEqual(filtered['concept_input'].shape, (2, 2))
        self.assertEqual(filtered['concept_target'].shape, (2, 2))
        self.assertEqual(filtered['task_input'].shape, (2, 1))
        self.assertEqual(filtered['task_input_with_rec'].shape, (2, 1))
        self.assertEqual(filtered['task_target'].shape, (2, 1))

    def test_filter_output_for_loss_requires_dict(self):
        """Test filter_output_for_loss rejects non-dict and incomplete dict inputs."""
        out = torch.randn(2, 3)
        target = torch.randint(0, 2, out.shape).float()

        with self.assertRaises(ValueError):
            self.model.filter_output_for_loss(out, target)

        with self.assertRaises(ValueError):
            self.model.filter_output_for_loss({'no_rec': out}, target)

    def test_filter_output_for_metrics(self):
        """Test filter_output_for_metrics returns correct format."""
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task1']
        out = self.model(query=query, x=x)
        target = torch.randint(0, 2, out.shape).float()

        filtered = self.model.filter_output_for_metrics(out, target)

        self.assertIsInstance(filtered, dict)
        self.assertIn('preds', filtered)
        self.assertIn('target', filtered)
        self.assertTrue(torch.allclose(filtered['preds'], out))
        self.assertTrue(torch.allclose(filtered['target'], target))


if __name__ == '__main__':
    unittest.main()
