"""
Comprehensive tests for torch_concepts.nn.modules.loss

Tests loss functions for concept-based learning:
- ConceptLoss: Unified loss for concepts with different types
- WeightedConceptLoss: Weighted combination of concept and task losses
- DepthWeightedConceptLoss: Graph-depth-weighted concept losses
"""
import unittest
import torch
from torch import nn
from torch_concepts.nn.modules.loss import ConceptLoss, WeightedConceptLoss, DepthWeightedConceptLoss
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
        loss_fn = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        
        # Binary concepts: endogenous shape (batch, 3)
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_categorical_only_loss(self):
        """Test ConceptLoss with only categorical concepts."""
        loss_fn = ConceptLoss(self.annotations_categorical, categorical=nn.CrossEntropyLoss())
        
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
        loss_fn = ConceptLoss(
            self.annotations_mixed,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        
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
        loss_fn = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        
        endogenous = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(endogenous, targets)
        loss.backward()
        
        self.assertIsNotNone(endogenous.grad)
        self.assertTrue(torch.any(endogenous.grad != 0))

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def test_repr_binary_only(self):
        """repr shows binary loss type."""
        loss = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        r = repr(loss)
        self.assertIn('ConceptLoss', r)
        self.assertIn('binary=BCEWithLogitsLoss', r)

    def test_repr_mixed(self):
        """repr shows all configured loss types."""
        loss = ConceptLoss(
            self.annotations_mixed,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss(),
        )
        r = repr(loss)
        self.assertIn('binary=BCEWithLogitsLoss', r)
        self.assertIn('categorical=CrossEntropyLoss', r)

    # ------------------------------------------------------------------
    # Continuous concepts (guarded by check_collection)
    # ------------------------------------------------------------------
    def test_continuous_raises_at_construction(self):
        """Continuous concepts raise NotImplementedError via check_collection."""
        axis = AxisAnnotation(
            labels=('cont1',),
            cardinalities=[1],
            metadata={'cont1': {'type': 'continuous'}}
        )
        ann = Annotations({1: axis})
        with self.assertRaises(NotImplementedError):
            ConceptLoss(ann, continuous=nn.MSELoss())


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
        loss_fn = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.5,
            task_weight=0.5,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        # 5 binary concepts total (3 concepts + 2 tasks)
        endogenous = torch.randn(16, 5)
        targets = torch.randint(0, 2, (16, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_concept_only_weight(self):
        """Test with concept_weight=1.0, task_weight=0.0 (only concept loss)."""
        loss_fn = WeightedConceptLoss(
            self.annotations,
            concept_weight=1.0,
            task_weight=0.0,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        self.assertTrue(loss >= 0)

    def test_task_only_weight(self):
        """Test with concept_weight=0.0, task_weight=1.0 (only task loss)."""
        loss_fn = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.0,
            task_weight=1.0,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        self.assertTrue(loss >= 0)

    def test_different_weights(self):
        """Test that different weights produce different losses."""
        torch.manual_seed(42)
        endogenous = torch.randn(20, 5)
        targets = torch.randint(0, 2, (20, 5)).float()
        
        loss_fn_high_concept = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.9,
            task_weight=0.1,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_fn_high_task = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.1,
            task_weight=0.9,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        loss_high_concept = loss_fn_high_concept(endogenous, targets)
        loss_high_task = loss_fn_high_task(endogenous, targets)
        
        # Losses should be different
        self.assertNotAlmostEqual(loss_high_concept.item(), loss_high_task.item(), places=3)

    def test_mixed_concept_types(self):
        """Test with mixed concept types (binary and categorical)."""
        loss_fn = WeightedConceptLoss(
            self.annotations_mixed,
            concept_weight=0.6,
            task_weight=0.4,
            task_names=self.task_names_mixed,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
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
        loss_fn = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.5,
            task_weight=0.5,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        
        endogenous = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 2, (8, 5)).float()
        
        loss = loss_fn(endogenous, targets)
        loss.backward()
        
        self.assertIsNotNone(endogenous.grad)
        self.assertTrue(torch.any(endogenous.grad != 0))

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def test_repr(self):
        """repr includes class name and fn_collection."""
        loss_fn = WeightedConceptLoss(
            self.annotations,
            concept_weight=0.5,
            task_weight=0.5,
            task_names=self.task_names,
            binary=nn.BCEWithLogitsLoss()
        )
        r = repr(loss_fn)
        self.assertIn('WeightedConceptLoss', r)
        self.assertIn('fn_collection', r)

    def test_weight_range(self):
        """Test various weight values in valid range [0, 1]."""
        endogenous = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()
        
        for concept_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            task_weight = 1.0 - concept_weight
            loss_fn = WeightedConceptLoss(
                self.annotations,
                concept_weight=concept_weight,
                task_weight=task_weight,
                task_names=self.task_names,
                binary=nn.BCEWithLogitsLoss()
            )
            
            loss = loss_fn(endogenous, targets)
            self.assertTrue(loss >= 0, f"Loss should be non-negative for concept_weight={concept_weight}")


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
        with self.assertRaises(ValueError):
            ConceptLoss(annotations, categorical=nn.CrossEntropyLoss())

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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConceptLoss(annotations, binary=nn.BCEWithLogitsLoss(), continuous=nn.MSELoss())
            # Should warn about unused continuous loss
            self.assertTrue(any("continuous" in str(warning.message).lower() for warning in w))


# ======================================================================
# DepthWeightedConceptLoss tests
# ======================================================================

from torch_concepts.nn.modules.mid.constructors.concept_graph import ConceptGraph


class TestDepthWeightedConceptLoss(unittest.TestCase):
    """Test DepthWeightedConceptLoss for graph-depth-based weighting."""

    def setUp(self):
        """Set up test fixtures.

        Graph:  A -> B -> C   (depths: A=0, B=1, C=2)
        All binary concepts.
        """
        from torch.distributions import Bernoulli

        self.axis = AxisAnnotation(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        self.annotations = Annotations({1: self.axis})

        adj = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])
        self.graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def test_basic_construction(self):
        """DepthWeightedConceptLoss initialises without error."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        self.assertIsInstance(loss_fn, nn.Module)

    def test_depth_levels_detected(self):
        """Three distinct depth levels are detected for A->B->C."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        self.assertEqual(loss_fn._depth_levels, [0, 1, 2])

    def test_depth_weights(self):
        """Weights follow source_weight * depth_decay ** d."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=2.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        expected = [2.0, 1.0, 0.5]  # 2*0.5^0, 2*0.5^1, 2*0.5^2
        for actual, exp in zip(loss_fn._depth_weights_list, expected):
            self.assertAlmostEqual(actual, exp)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def test_forward_returns_scalar(self):
        """Forward returns a scalar loss."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            binary=nn.BCEWithLogitsLoss()
        )
        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        loss = loss_fn(preds, targets)

        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_forward_weighting_effect(self):
        """Higher source_weight produces proportionally larger loss."""
        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        loss_fn_1 = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=1.0, depth_decay=1.0,
            binary=nn.BCEWithLogitsLoss()
        )
        loss_fn_2 = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=2.0, depth_decay=1.0,
            binary=nn.BCEWithLogitsLoss()
        )
        loss1 = loss_fn_1(preds, targets)
        loss2 = loss_fn_2(preds, targets)
        self.assertTrue(torch.allclose(loss2, 2.0 * loss1, atol=1e-5))

    def test_depth_decay_down_weights_deeper(self):
        """With decay < 1 the root concept contributes more than deeper ones."""
        # Use fixed predictions so we can reason about relative magnitudes
        torch.manual_seed(0)
        preds = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()

        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=1.0, depth_decay=0.01,  # heavily down-weight deeper
            binary=nn.BCEWithLogitsLoss()
        )
        loss = loss_fn(preds, targets)

        # Compare with loss coming only from root (depth 0)
        root_ann = Annotations({1: self.axis.subset(['A'])})
        root_loss_fn = ConceptLoss(root_ann, binary=nn.BCEWithLogitsLoss())
        root_loss = root_loss_fn(preds[:, 0:1], targets[:, 0:1])

        # Loss should be dominated by root; difference from root should be small
        relative_diff = (loss - root_loss).abs() / root_loss
        self.assertTrue(relative_diff < 0.05)

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------
    def test_gradient_flow(self):
        """Gradients flow through every depth level."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            binary=nn.BCEWithLogitsLoss()
        )
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()

        loss = loss_fn(preds, targets)
        loss.backward()

        self.assertIsNotNone(preds.grad)
        # All three columns should receive gradients (one per depth)
        for col in range(3):
            self.assertTrue(torch.any(preds.grad[:, col] != 0),
                            f"No gradient for column {col}")

    # ------------------------------------------------------------------
    # Mixed concept types
    # ------------------------------------------------------------------
    def test_mixed_binary_categorical(self):
        """Works with a mix of binary and categorical concepts."""
        from torch.distributions import Bernoulli, Categorical

        axis = AxisAnnotation(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 3, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Categorical},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = Annotations({1: axis})
        adj = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])

        loss_fn = DepthWeightedConceptLoss(
            ann, graph,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        # logit dim = 1 + 3 + 1 = 5
        preds = torch.randn(8, 5)
        targets = torch.cat([
            torch.randint(0, 2, (8, 1)).float(),
            torch.randint(0, 3, (8, 1)).float(),
            torch.randint(0, 2, (8, 1)).float(),
        ], dim=1)
        loss = loss_fn(preds, targets)
        self.assertEqual(loss.shape, ())

    # ------------------------------------------------------------------
    # Multiple roots / diamond graph
    # ------------------------------------------------------------------
    def test_diamond_graph(self):
        """Diamond graph: two roots converge at a single leaf.

        Graph:  A -->  C
                B -->  C
        Depths: A=0, B=0, C=1
        """
        from torch.distributions import Bernoulli

        axis = AxisAnnotation(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = Annotations({1: axis})
        adj = torch.tensor([
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])

        loss_fn = DepthWeightedConceptLoss(
            ann, graph,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        # Two depth levels: 0 and 1
        self.assertEqual(loss_fn._depth_levels, [0, 1])
        self.assertAlmostEqual(loss_fn._depth_weights_list[0], 1.0)
        self.assertAlmostEqual(loss_fn._depth_weights_list[1], 0.5)

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        loss = loss_fn(preds, targets)
        self.assertEqual(loss.shape, ())

    # ------------------------------------------------------------------
    # Concept not in graph defaults to depth 0
    # ------------------------------------------------------------------
    def test_concept_not_in_graph_defaults_depth_zero(self):
        """Concepts absent from the graph are treated as depth 0."""
        from torch.distributions import Bernoulli

        # Graph only contains A and B
        adj_small = torch.tensor([
            [0., 1.],
            [0., 0.],
        ])
        graph_small = ConceptGraph(adj_small, node_names=['A', 'B'])

        loss_fn = DepthWeightedConceptLoss(
            self.annotations, graph_small,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        # C not in graph → depth 0; A depth 0; B depth 1
        # So depth 0 has [A, C], depth 1 has [B]
        self.assertEqual(loss_fn._depth_levels, [0, 1])

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def test_repr(self):
        """repr includes class name and depth/weight information."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        r = repr(loss_fn)
        self.assertIn('DepthWeightedConceptLoss', r)
        self.assertIn('depth_0', r)
        self.assertIn('depth_1', r)
        self.assertIn('depth_2', r)

    # ------------------------------------------------------------------
    # Sub-modules visible
    # ------------------------------------------------------------------
    def test_submodules_visible(self):
        """Sub-ConceptLoss modules are visible as named_modules."""
        loss_fn = DepthWeightedConceptLoss(
            self.annotations, self.graph,
            binary=nn.BCEWithLogitsLoss()
        )
        names = [name for name, _ in loss_fn.named_modules()]
        self.assertTrue(any('loss_depth_0' in n for n in names))
        self.assertTrue(any('loss_depth_1' in n for n in names))
        self.assertTrue(any('loss_depth_2' in n for n in names))

    def test_missing_concepts_creates_new_depth_zero(self):
        """When no graph node overlaps with annotations, missing branch creates depth_0."""
        from torch.distributions import Bernoulli

        # Graph nodes X->Y don't appear in the annotations at all
        adj = torch.tensor([
            [0., 1.],
            [0., 0.],
        ])
        graph_xy = ConceptGraph(adj, node_names=['X', 'Y'])

        # Annotations only have A, B — neither in the graph
        axis = AxisAnnotation(
            labels=['A', 'B'],
            cardinalities=[1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = Annotations({1: axis})

        loss_fn = DepthWeightedConceptLoss(
            ann, graph_xy,
            source_weight=1.0, depth_decay=0.5,
            binary=nn.BCEWithLogitsLoss()
        )
        # Both A, B are missing from graph → assigned to depth 0
        self.assertIn(0, loss_fn._depth_levels)
        self.assertTrue(hasattr(loss_fn, 'loss_depth_0'))
        # Forward should work
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        loss = loss_fn(preds, targets)
        self.assertEqual(loss.shape, ())


# ======================================================================
# L1LogitRegularizer tests
# ======================================================================
from torch_concepts.nn.modules.loss import L1LogitRegularizer


class TestL1LogitRegularizer(unittest.TestCase):
    """Test L1LogitRegularizer."""

    def test_default_scale(self):
        """Default scale=1.0 returns mean absolute value."""
        reg = L1LogitRegularizer()
        self.assertEqual(reg.scale, 1.0)
        x = torch.tensor([1.0, -2.0, 3.0, -4.0])
        expected = x.abs().mean()
        self.assertTrue(torch.allclose(reg(x), expected))

    def test_custom_scale(self):
        """Custom scale multiplies the L1 mean."""
        reg = L1LogitRegularizer(scale=0.5)
        x = torch.tensor([2.0, -4.0])
        expected = 0.5 * x.abs().mean()
        self.assertTrue(torch.allclose(reg(x), expected))

    def test_zero_input(self):
        """Zero logits produce zero loss."""
        reg = L1LogitRegularizer(scale=2.0)
        x = torch.zeros(5)
        self.assertEqual(reg(x).item(), 0.0)

    def test_gradient_flow(self):
        """Gradients flow through the regularizer."""
        reg = L1LogitRegularizer()
        x = torch.randn(8, requires_grad=True)
        loss = reg(x)
        loss.backward()
        self.assertIsNotNone(x.grad)


# ======================================================================
# CompositeLoss tests
# ======================================================================

from functools import partial
from torch_concepts.nn.modules.loss import CompositeLoss
from torch_concepts.nn.modules.utils import GroupConfig


class _DummyInputOnlyLoss(nn.Module):
    """A toy loss that only uses ``input`` (e.g. an L1 regulariser on logits)."""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.abs().mean()


class _DummyExtraKwargLoss(nn.Module):
    """A toy loss that needs an extra kwarg ``embeddings``."""
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.norm(dim=-1).mean()


class _DummyVarKwargsLoss(nn.Module):
    """A loss whose forward accepts **kwargs (receives everything)."""
    def forward(self, **kwargs) -> torch.Tensor:
        total = torch.tensor(0.0)
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                total = total + v.abs().mean()
        return total


class TestCompositeLoss(unittest.TestCase):
    """Test CompositeLoss for modular loss composition."""

    def setUp(self):
        """Set up test fixtures."""
        axis = AxisAnnotation(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations = Annotations({1: axis})
        self.fn_collection = GroupConfig(binary=nn.BCEWithLogitsLoss())

    # ------------------------------------------------------------------
    # Basic construction
    # ------------------------------------------------------------------
    def test_single_term_no_weight(self):
        """A single term with default weight=1.0 behaves like the original loss."""
        concept_loss = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        composite = CompositeLoss(terms=[concept_loss])

        self.assertEqual(len(composite.terms), 1)
        self.assertEqual(composite.weights, [1.0])

    def test_multiple_terms_with_weights(self):
        """Multiple terms with explicit weights."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()
        composite = CompositeLoss(terms=[t1, t2], weights=[1.0, 0.5])

        self.assertEqual(len(composite.terms), 2)
        self.assertEqual(composite.weights, [1.0, 0.5])

    def test_weight_count_mismatch_raises(self):
        """Mismatched weight/term count should raise ValueError."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        with self.assertRaises(ValueError):
            CompositeLoss(terms=[t1], weights=[1.0, 2.0])

    def test_non_module_non_callable_raises(self):
        """A plain string or int in terms should raise TypeError."""
        with self.assertRaises(TypeError):
            CompositeLoss(terms=["not_a_loss"])

    # ------------------------------------------------------------------
    # Forward / kwarg dispatch
    # ------------------------------------------------------------------
    def test_forward_single_concept_loss(self):
        """Single ConceptLoss through CompositeLoss matches direct call."""
        concept_loss = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        composite = CompositeLoss(terms=[concept_loss])

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        direct = concept_loss(preds, targets)
        via_composite = composite(input=preds, target=targets)

        self.assertTrue(torch.allclose(direct, via_composite))

    def test_forward_weighted_sum(self):
        """Weighted sum is correct numerically."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        w1, w2 = 2.0, 0.5
        composite = CompositeLoss(terms=[t1, t2], weights=[w1, w2])

        expected = w1 * t1(preds, targets) + w2 * t2(preds)
        actual = composite(input=preds, target=targets)

        self.assertTrue(torch.allclose(expected, actual))

    def test_extra_kwargs_dispatched(self):
        """Extra kwargs (embeddings) are dispatched to the right term only."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyExtraKwargLoss()

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        emb = torch.randn(8, 16)

        composite = CompositeLoss(terms=[t1, t2], weights=[1.0, 0.1])

        # Should not error — t1 gets (input, target), t2 gets (embeddings,)
        loss = composite(input=preds, target=targets, embeddings=emb)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())

    def test_var_kwargs_term_receives_all(self):
        """A term with **kwargs in forward receives all kwargs."""
        t1 = _DummyVarKwargsLoss()
        composite = CompositeLoss(terms=[t1])

        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        loss = composite(input=preds, target=targets)
        self.assertTrue(loss > 0)

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------
    def test_gradient_flow_through_composite(self):
        """Gradients flow through every term."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()

        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()

        composite = CompositeLoss(terms=[t1, t2], weights=[1.0, 0.5])
        loss = composite(input=preds, target=targets)
        loss.backward()

        self.assertIsNotNone(preds.grad)
        self.assertTrue(torch.any(preds.grad != 0))

    # ------------------------------------------------------------------
    # Partial / callable resolution (Hydra-style)
    # ------------------------------------------------------------------
    def test_partial_term_resolved_with_annotations(self):
        """functools.partial terms are finalised with common_kwargs."""
        partial_concept_loss = partial(
            ConceptLoss,
            binary=nn.BCEWithLogitsLoss(),
        )
        # annotations passed via common_kwargs
        composite = CompositeLoss(
            terms=[partial_concept_loss],
            annotations=self.annotations,
        )
        self.assertEqual(len(composite.terms), 1)
        self.assertIsInstance(list(composite.terms.values())[0], ConceptLoss)

    def test_mixed_partial_and_module(self):
        """Mix of partial and already-instantiated terms works."""
        partial_concept_loss = partial(
            ConceptLoss,
            binary=nn.BCEWithLogitsLoss(),
        )
        reg = _DummyInputOnlyLoss()

        composite = CompositeLoss(
            terms=[partial_concept_loss, reg],
            weights=[1.0, 0.3],
            annotations=self.annotations,
        )

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        loss = composite(input=preds, target=targets)
        self.assertIsInstance(loss, torch.Tensor)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def test_repr(self):
        """repr is human-readable."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()
        composite = CompositeLoss(terms=[t1, t2], weights=[1.0, 0.5])
        r = repr(composite)
        self.assertIn("CompositeLoss", r)
        self.assertIn("ConceptLoss", r)
        self.assertIn("0.5", r)

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------
    def test_device_from_kwargs(self):
        """Scalar accumulator moves to correct device (CPU in CI)."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        composite = CompositeLoss(terms=[t1])

        preds = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        loss = composite(input=preds, target=targets)
        self.assertEqual(loss.device, preds.device)

    # ------------------------------------------------------------------
    # forward_detailed
    # ------------------------------------------------------------------
    def test_forward_detailed_returns_per_term_losses(self):
        """forward_detailed returns total and per-term dict."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        w1, w2 = 1.0, 0.5
        composite = CompositeLoss(terms=[t1, t2], weights=[w1, w2])

        total, details = composite.forward_detailed(input=preds, target=targets)

        # Total matches forward()
        expected_total = composite(input=preds, target=targets)
        self.assertTrue(torch.allclose(total, expected_total))

        # Per-term keys match class names
        self.assertIn('ConceptLoss', details)
        self.assertIn('_DummyInputOnlyLoss', details)

        # Weighted values sum to total
        detail_sum = sum(details.values())
        self.assertTrue(torch.allclose(total, detail_sum))

    def test_forward_detailed_single_term(self):
        """forward_detailed with one term: total == single term loss."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        composite = CompositeLoss(terms=[t1])

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        total, details = composite.forward_detailed(input=preds, target=targets)
        self.assertEqual(len(details), 1)
        self.assertTrue(torch.allclose(total, details['ConceptLoss']))

    # ------------------------------------------------------------------
    # Module dict keys / model summary
    # ------------------------------------------------------------------
    def test_term_keys_unique_for_duplicate_types(self):
        """Two terms of the same class get distinct keys."""
        t1 = _DummyInputOnlyLoss()
        t2 = _DummyInputOnlyLoss()
        composite = CompositeLoss(terms=[t1, t2])

        keys = list(composite.terms.keys())
        self.assertEqual(keys, ['_DummyInputOnlyLoss', '_DummyInputOnlyLoss_1'])

    def test_terms_visible_in_named_modules(self):
        """Individual terms appear as named sub-modules (visible in model summary)."""
        t1 = ConceptLoss(self.annotations, binary=nn.BCEWithLogitsLoss())
        t2 = _DummyInputOnlyLoss()
        composite = CompositeLoss(terms=[t1, t2])

        module_names = [name for name, _ in composite.named_modules()]
        # e.g. '', 'terms', 'terms.ConceptLoss', 'terms._DummyInputOnlyLoss'
        self.assertTrue(any('ConceptLoss' in n for n in module_names))
        self.assertTrue(any('_DummyInputOnlyLoss' in n for n in module_names))

    # ------------------------------------------------------------------
    # Callable fallback (inspect.signature raises)
    # ------------------------------------------------------------------
    def test_callable_fallback_on_signature_error(self):
        """Callable term whose signature cannot be inspected falls back to passing all common_kwargs."""
        from unittest.mock import patch
        import inspect as _inspect

        class _SimpleLoss(nn.Module):
            def forward(self, input, target):
                return (input - target).abs().mean()

        # A factory that returns an nn.Module
        def factory(**kwargs):
            return _SimpleLoss()

        original_sig = _inspect.signature

        def _patched_sig(obj, **kwargs):
            if obj is factory:
                raise ValueError("no sig")
            return original_sig(obj, **kwargs)

        with patch.object(_inspect, 'signature', _patched_sig):
            composite = CompositeLoss(terms=[factory])

        self.assertEqual(len(composite.terms), 1)
        self.assertIsInstance(list(composite.terms.values())[0], _SimpleLoss)

    # ------------------------------------------------------------------
    # _zero with no tensor kwargs
    # ------------------------------------------------------------------
    def test_zero_no_tensor_kwargs(self):
        """_zero returns CPU tensor when kwargs contain no tensors."""
        result = CompositeLoss._zero({'a': 'string', 'b': 42})
        self.assertEqual(result.item(), 0.0)
        self.assertEqual(result.device, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
