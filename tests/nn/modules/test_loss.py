"""
Comprehensive tests for torch_concepts.nn.modules.loss

Tests loss functions for concept-based learning:
- ConceptLoss: Unified loss for concepts with different types
- WeightedConceptLoss: Weighted combination of concept and task losses
"""
import unittest
import torch
from torch import nn
from torch_concepts.nn.modules.loss import ConceptLoss, WeightedConceptLoss
from torch_concepts.nn.modules.utils import GroupConfig
from torch_concepts.annotations import AxisAnnotation, Annotations


class TestConceptLoss(unittest.TestCase):
    """Test ConceptLoss for unified concept loss computation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create annotations with mixed concept types (binary and categorical only)
        axis_mixed = AxisAnnotation(
            labels=('binary1', 'binary2', 'cat1', 'cat2'),
            cardinalities=[1, 1, 3, 4],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = Annotations({1: axis_mixed})
        
        # All binary
        axis_binary = AxisAnnotation(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations_binary = Annotations({1: axis_binary})
        
        # All categorical
        axis_categorical = AxisAnnotation(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_categorical = Annotations({1: axis_categorical})
        
        # All continuous - not currently tested as continuous concepts are not fully supported
        # self.annotations_continuous = AxisAnnotation(
        #     labels=('cont1', 'cont2', 'cont3'),
        #     cardinalities=(1, 1, 1),
        #     metadata={
        #         'cont1': {'type': 'continuous'},
        #         'cont2': {'type': 'continuous'},
        #         'cont3': {'type': 'continuous'},
        #     }
        # )

    def test_binary_only_loss(self):
        """Test ConceptLoss with only binary concepts."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = ConceptLoss(self.annotations_binary, loss_config)
        
        # Binary concepts: endogenous shape (batch, 3)
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_categorical_only_loss(self):
        """Test ConceptLoss with only categorical concepts."""
        loss_config = GroupConfig(
            categorical=nn.CrossEntropyLoss()
        )
        
        loss_fn = ConceptLoss(self.annotations_categorical, loss_config)
        
        # Categorical: cat1 (3 classes) + cat2 (5 classes) = 8 endogenous total
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    # Continuous concepts are not fully supported yet - skipping test
    # def test_continuous_only_loss(self):
    #     """Test ConceptLoss with only continuous concepts."""
    #     pass

    def test_mixed_concepts_loss(self):
        """Test ConceptLoss with mixed concept types (binary and categorical only)."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        
        loss_fn = ConceptLoss(self.annotations_mixed, loss_config)
        
        # Mixed: 2 binary + (3 + 4) categorical = 9 endogenous
        endogenous = torch.randn(16, 9)
        targets = torch.cat([
            torch.randint(0, 2, (16, 2)).float(),  # binary
            torch.randint(0, 3, (16, 1)),  # cat1
            torch.randint(0, 4, (16, 1)),  # cat2
        ], dim=1)
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow properly through ConceptLoss."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = ConceptLoss(self.annotations_binary, loss_config)
        
        endogenous = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(endogenous, targets)
        loss.backward()
        
        self.assertIsNotNone(endogenous.grad)
        self.assertTrue(torch.any(endogenous.grad != 0))

    # Continuous concepts are not fully supported yet - skipping tests
    # def test_perfect_predictions(self):
    #     """Test with perfect continuous predictions (near-zero loss)."""
    #     pass
    
    # def test_multidim_continuous_concepts(self):
    #     """Test ConceptLoss with multi-dimensional continuous concepts."""
    #     pass


class TestWeightedConceptLoss(unittest.TestCase):
    """Test WeightedConceptLoss for weighted concept and task losses."""

    def setUp(self):
        """Set up test fixtures."""
        # Create annotations with concepts and tasks
        self.annotations = AxisAnnotation(
            labels=('concept1', 'concept2', 'concept3', 'task1', 'task2'),
            cardinalities=(1, 1, 1, 1, 1),
            metadata={
                'concept1': {'type': 'discrete'},
                'concept2': {'type': 'discrete'},
                'concept3': {'type': 'discrete'},
                'task1': {'type': 'discrete'},
                'task2': {'type': 'discrete'},
            }
        )
        self.annotations = Annotations({1: self.annotations})
        
        self.task_names = ['task1', 'task2']
        
        # Mixed types (binary and categorical only - continuous not supported yet)
        self.annotations_mixed = AxisAnnotation(
            labels=('c1', 'c2', 'c3', 't1', 't2'),
            cardinalities=(1, 3, 1, 1, 4),
            metadata={
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'},
                'c3': {'type': 'discrete'},
                't1': {'type': 'discrete'},
                't2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = Annotations({1: self.annotations_mixed})
        
        self.task_names_mixed = ['t1', 't2']

    def test_basic_forward(self):
        """Test basic forward pass with balanced weighting."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = WeightedConceptLoss(
            self.annotations, 
            loss_config, 
            weight=0.5,
            task_names=self.task_names
        )
        
        # 5 binary concepts total (3 concepts + 2 tasks)
        endogenous = torch.randn(16, 5)
        targets = torch.randint(0, 2, (16, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_concept_only_weight(self):
        """Test with weight=1.0 (only concept loss)."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = WeightedConceptLoss(
            self.annotations,
            loss_config,
            weight=1.0,
            task_names=self.task_names
        )
        
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        self.assertTrue(loss >= 0)

    def test_task_only_weight(self):
        """Test with weight=0.0 (only task loss)."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = WeightedConceptLoss(
            self.annotations,
            loss_config,
            weight=0.0,
            task_names=self.task_names
        )
        
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        self.assertTrue(loss >= 0)

    def test_different_weights(self):
        """Test that different weights produce different losses."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        torch.manual_seed(42)
        endogenous = torch.randn(20, 5)
        targets = torch.randint(0, 2, (20, 5)).float()
        
        loss_fn_high_concept = WeightedConceptLoss(
            self.annotations,
            loss_config,
            weight=0.9,
            task_names=self.task_names
        )
        
        loss_fn_high_task = WeightedConceptLoss(
            self.annotations,
            loss_config,
            weight=0.1,
            task_names=self.task_names
        )
        
        loss_high_concept = loss_fn_high_concept(endogenous, targets)
        loss_high_task = loss_fn_high_task(endogenous, targets)
        
        # Losses should be different
        self.assertNotAlmostEqual(loss_high_concept.item(), loss_high_task.item(), places=3)

    def test_mixed_concept_types(self):
        """Test with mixed concept types (binary and categorical)."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        
        loss_fn = WeightedConceptLoss(
            self.annotations_mixed,
            loss_config,
            weight=0.6,
            task_names=self.task_names_mixed
        )
        
        # c1 (1) + c2 (3) + c3 (1) + t1 (1) + t2 (4) = 10 endogenous
        endogenous = torch.randn(16, 10)
        targets = torch.cat([
            torch.randint(0, 2, (16, 1)).float(),  # c1 binary
            torch.randint(0, 3, (16, 1)),  # c2 categorical
            torch.randint(0, 2, (16, 1)).float(),  # c3 binary
            torch.randint(0, 2, (16, 1)).float(),  # t1 binary
            torch.randint(0, 4, (16, 1)),  # t2 categorical
        ], dim=1)
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow properly through WeightedConceptLoss."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn = WeightedConceptLoss(
            self.annotations,
            loss_config,
            weight=0.5,
            task_names=self.task_names
        )
        
        endogenous = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 2, (8, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        loss.backward()
        
        self.assertIsNotNone(endogenous.grad)
        self.assertTrue(torch.any(endogenous.grad != 0))

    def test_weight_range(self):
        """Test various weight values in valid range [0, 1]."""
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss()
        )
        
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss_fn = WeightedConceptLoss(
                self.annotations,
                loss_config,
                weight=weight,
                task_names=self.task_names
            )
            
            loss = loss_fn(endogenous, targets)
            self.assertTrue(loss >= 0, f"Loss should be non-negative for weight={weight}")


class TestLossConfiguration(unittest.TestCase):
    """Test loss configuration and setup."""

    def test_missing_required_loss_config(self):
        """Test that missing required loss config raises error."""
        axis = AxisAnnotation(
            labels=('b1', 'b2'),
            cardinalities=(1, 1),
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
            }
        )
        annotations = Annotations({1: axis})
        
        # Missing binary loss config (only provides categorical)
        loss_config = GroupConfig(
            categorical=nn.CrossEntropyLoss()
        )
        
        with self.assertRaises(ValueError):
            ConceptLoss(annotations, loss_config)

    def test_unused_loss_warning(self):
        """Test that unused loss configs produce warnings."""
        import warnings
        
        axis = AxisAnnotation(
            labels=('b1', 'b2'),
            cardinalities=(1, 1),
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
            }
        )
        annotations = Annotations({1: axis})
        
        # Provides continuous loss but no continuous concepts
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss(),
            continuous=nn.MSELoss()
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConceptLoss(annotations, loss_config)
            # Should warn about unused continuous loss
            self.assertTrue(any("continuous" in str(warning.message).lower() for warning in w))


if __name__ == '__main__':
    unittest.main()
