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
from torch_concepts.nn.modules.loss import ConceptLoss, WeightedConceptLoss, DepthWeightedConceptLoss, L1LogitRegularizer
from torch_concepts.nn.modules.outputs import ModelOutput
from torch_concepts.annotations import Annotations


class TestConceptLoss(unittest.TestCase):
    """Test ConceptLoss for unified concept loss computation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create annotations with mixed concept types (binary and categorical only)
        axis_mixed = Annotations(
            labels=('binary1', 'binary2', 'cat1', 'cat2'),
            cardinalities=[1, 1, 3, 4],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = axis_mixed
        
        # All binary
        axis_binary = Annotations(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations_binary = axis_binary
        
        # All categorical
        axis_categorical = Annotations(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_categorical = axis_categorical
        
        # All continuous - not currently tested as continuous concepts are not fully supported
        # self.annotations_continuous = Annotations(
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow properly through ConceptLoss."""
        loss_fn = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        
        endogenous = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
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
        axis = Annotations(
            labels=('cont1',),
            cardinalities=[1],
            types=['continuous'],
        )
        ann = axis
        with self.assertRaises(NotImplementedError):
            ConceptLoss(ann, continuous=nn.MSELoss())


class TestWeightedConceptLoss(unittest.TestCase):
    """Test WeightedConceptLoss for weighted concept and task losses."""

    def setUp(self):
        """Set up test fixtures."""
        # Create annotations with concepts and tasks
        self.annotations = Annotations(
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
        self.annotations = self.annotations
        
        self.task_names = ['task1', 'task2']
        
        # Mixed types (binary and categorical only - continuous not supported yet)
        self.annotations_mixed = Annotations(
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
        self.annotations_mixed = self.annotations_mixed
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
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
        
        loss_high_concept = loss_fn_high_concept(ModelOutput(logits=endogenous, target=targets))
        loss_high_task = loss_fn_high_task(ModelOutput(logits=endogenous, target=targets))
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
        
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
        
        loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
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
            
            loss = loss_fn(ModelOutput(logits=endogenous, target=targets))
            self.assertTrue(loss >= 0, f"Loss should be non-negative for concept_weight={concept_weight}")


class TestLossConfiguration(unittest.TestCase):
    """Test loss configuration and setup."""

    def test_missing_required_loss_config(self):
        """Test that missing required loss config raises error."""
        axis = Annotations(
            labels=('b1', 'b2'),
            cardinalities=(1, 1),
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
            }
        )
        annotations = axis
        
        # Missing binary loss config (only provides categorical)
        with self.assertRaises(ValueError):
            ConceptLoss(annotations, categorical=nn.CrossEntropyLoss())

    def test_unused_loss_warning(self):
        """Test that unused loss configs produce warnings."""
        import warnings
        
        axis = Annotations(
            labels=('b1', 'b2'),
            cardinalities=(1, 1),
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
            }
        )
        annotations = axis
        
        # Provides continuous loss but no continuous concepts
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConceptLoss(annotations, binary=nn.BCEWithLogitsLoss(), continuous=nn.MSELoss())
            # Should warn about unused continuous loss
            self.assertTrue(any("continuous" in str(warning.message).lower() for warning in w))


# ======================================================================
# DepthWeightedConceptLoss tests
# ======================================================================

from torch_concepts import ConceptGraph


class TestDepthWeightedConceptLoss(unittest.TestCase):
    """Test DepthWeightedConceptLoss for graph-depth-based weighting."""

    def setUp(self):
        """Set up test fixtures.

        Graph:  A -> B -> C   (depths: A=0, B=1, C=2)
        All binary concepts.
        """
        from torch.distributions import Bernoulli

        self.axis = Annotations(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        self.annotations = self.axis

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
        loss = loss_fn(ModelOutput(logits=preds, target=targets))

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
        loss1 = loss_fn_1(ModelOutput(logits=preds, target=targets))
        loss2 = loss_fn_2(ModelOutput(logits=preds, target=targets))
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
        loss = loss_fn(ModelOutput(logits=preds, target=targets))

        # Compare with loss coming only from root (depth 0)
        root_ann = self.axis.subset(['A'])
        root_loss_fn = ConceptLoss(root_ann, binary=nn.BCEWithLogitsLoss())
        root_loss = root_loss_fn(ModelOutput(logits=preds[:, 0:1], target=targets[:, 0:1]))
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

        loss = loss_fn(ModelOutput(logits=preds, target=targets))
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
        from torch.distributions import Bernoulli, OneHotCategorical

        axis = Annotations(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 3, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': OneHotCategorical},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = axis
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
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
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

        axis = Annotations(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = axis
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
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
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
        axis = Annotations(
            labels=['A', 'B'],
            cardinalities=[1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
            }
        )
        ann = axis

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
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
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
# ConceptLoss composite per-type tests
# ======================================================================

class TestConceptLossComposite(unittest.TestCase):
    """Test ConceptLoss with list-based per-type loss composition."""

    def setUp(self):
        """Set up test fixtures."""
        axis_binary = Annotations(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations_binary = axis_binary

        axis_mixed = Annotations(
            labels=('binary1', 'binary2', 'cat1', 'cat2'),
            cardinalities=[1, 1, 3, 4],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = axis_mixed

    # ------------------------------------------------------------------
    # List construction
    # ------------------------------------------------------------------
    def test_binary_list_single_term(self):
        """A single-element list behaves like a single module."""
        loss_single = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        loss_list = ConceptLoss(self.annotations_binary, binary=[nn.BCEWithLogitsLoss()])

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        l1 = loss_single(ModelOutput(logits=preds, target=targets))
        l2 = loss_list(ModelOutput(logits=preds, target=targets))
        self.assertTrue(torch.allclose(l1, l2))

    def test_binary_list_with_regularizer(self):
        """Binary list with BCE + L1 regularizer produces expected loss."""
        bce = nn.BCEWithLogitsLoss()
        reg = L1LogitRegularizer(scale=0.01)

        loss_fn = ConceptLoss(
            self.annotations_binary,
            binary=[bce, reg],
            binary_weights=[1.0, 0.5],
        )

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        # Compute expected
        expected = 1.0 * bce(preds, targets) + 0.5 * reg(preds)
        actual = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertTrue(torch.allclose(expected, actual))

    def test_binary_list_default_weights(self):
        """Omitting weights defaults to [1.0, 1.0, ...]."""
        bce = nn.BCEWithLogitsLoss()
        reg = L1LogitRegularizer(scale=0.01)

        loss_fn = ConceptLoss(
            self.annotations_binary,
            binary=[bce, reg],
        )

        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()

        expected = bce(preds, targets) + reg(preds)
        actual = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertTrue(torch.allclose(expected, actual))

    def test_weight_count_mismatch_raises(self):
        """Mismatched weight/term count should raise ValueError."""
        with self.assertRaises(ValueError):
            ConceptLoss(
                self.annotations_binary,
                binary=[nn.BCEWithLogitsLoss()],
                binary_weights=[1.0, 0.5],
            )

    def test_mixed_composite_binary_only(self):
        """Composite on binary with single categorical in mixed annotations."""
        loss_fn = ConceptLoss(
            self.annotations_mixed,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
            categorical=nn.CrossEntropyLoss(),
        )

        # 2 binary + (3 + 4) categorical = 9 logits
        preds = torch.randn(16, 9)
        targets = torch.cat([
            torch.randint(0, 2, (16, 2)).float(),
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 4, (16, 1)),
        ], dim=1)

        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------
    def test_gradient_flow_composite(self):
        """Gradients flow through all terms in a composite list."""
        loss_fn = ConceptLoss(
            self.annotations_binary,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
        )

        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()

        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        loss.backward()

        self.assertIsNotNone(preds.grad)
        self.assertTrue(torch.any(preds.grad != 0))

    # ------------------------------------------------------------------
    # Module registration
    # ------------------------------------------------------------------
    def test_modules_registered(self):
        """Loss terms in lists are visible as sub-modules."""
        loss_fn = ConceptLoss(
            self.annotations_binary,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        )
        module_names = [name for name, _ in loss_fn.named_modules()]
        self.assertTrue(any('_binary_terms' in n for n in module_names))

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def test_repr_single(self):
        """Single module repr shows type=ClassName."""
        loss = ConceptLoss(self.annotations_binary, binary=nn.BCEWithLogitsLoss())
        r = repr(loss)
        self.assertIn('binary=BCEWithLogitsLoss', r)

    def test_repr_composite(self):
        """Composite repr shows weighted list."""
        loss = ConceptLoss(
            self.annotations_binary,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
        )
        r = repr(loss)
        self.assertIn('binary=', r)
        self.assertIn('BCEWithLogitsLoss', r)
        self.assertIn('L1LogitRegularizer', r)
        self.assertIn('0.5', r)


# ======================================================================
# Helper function tests
# ======================================================================

from torch_concepts.nn.modules.loss import _get_forward_signature, _normalize_loss_terms


class TestGetForwardSignature(unittest.TestCase):
    """Test _get_forward_signature introspection helper."""

    def test_standard_loss(self):
        sig, has_var_kw = _get_forward_signature(nn.BCEWithLogitsLoss())
        self.assertIn('input', sig)
        self.assertIn('target', sig)
        self.assertFalse(has_var_kw)

    def test_input_only(self):
        sig, has_var_kw = _get_forward_signature(L1LogitRegularizer())
        self.assertIn('input', sig)
        self.assertNotIn('target', sig)
        self.assertFalse(has_var_kw)

    def test_var_kwargs_detected(self):
        class _VarKwModule(nn.Module):
            def forward(self, **kwargs):
                return kwargs['input'].sum()
        sig, has_var_kw = _get_forward_signature(_VarKwModule())
        self.assertTrue(has_var_kw)

    def test_extra_param(self):
        class _ExtraModule(nn.Module):
            def forward(self, input, target, embeddings):
                return input.sum()
        sig, has_var_kw = _get_forward_signature(_ExtraModule())
        self.assertEqual(sig, {'input', 'target', 'embeddings'})
        self.assertFalse(has_var_kw)


class TestNormalizeLossTerms(unittest.TestCase):
    """Test _normalize_loss_terms helper."""

    def test_none_passthrough(self):
        terms, weights = _normalize_loss_terms(None, None)
        self.assertIsNone(terms)
        self.assertIsNone(weights)

    def test_single_module_wraps_to_list(self):
        m = nn.MSELoss()
        terms, weights = _normalize_loss_terms(m, None)
        self.assertEqual(len(terms), 1)
        self.assertIs(terms[0], m)
        self.assertEqual(weights, [1.0])

    def test_list_passthrough(self):
        m1, m2 = nn.MSELoss(), nn.L1Loss()
        terms, weights = _normalize_loss_terms([m1, m2], [0.8, 0.2])
        self.assertEqual(len(terms), 2)
        self.assertEqual(weights, [0.8, 0.2])

    def test_tuple_passthrough(self):
        terms, weights = _normalize_loss_terms((nn.MSELoss(),), None)
        self.assertEqual(len(terms), 1)
        self.assertEqual(weights, [1.0])

    def test_weight_count_mismatch(self):
        with self.assertRaises(ValueError):
            _normalize_loss_terms([nn.MSELoss()], [1.0, 2.0])

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _normalize_loss_terms("not_a_module", None)


# ======================================================================
# Extra-kwargs forwarding tests
# ======================================================================

class _EmbeddingAwareLoss(nn.Module):
    """Dummy loss that also uses an 'embeddings' kwarg."""
    def forward(self, input, target, embeddings):
        return nn.functional.mse_loss(input, target) + embeddings.abs().mean()


class _VarKwargsLoss(nn.Module):
    """Dummy loss with **kwargs — receives everything."""
    def forward(self, **kwargs):
        return kwargs['input'].abs().mean()


class TestConceptLossKwargsForwarding(unittest.TestCase):
    """Test that extra kwargs flow to loss terms based on signature."""

    def setUp(self):
        axis = Annotations(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={'b1': {'type': 'discrete'}, 'b2': {'type': 'discrete'}},
        )
        self.ann = axis

    def test_extra_kwarg_reaches_term(self):
        loss_fn = ConceptLoss(self.ann, binary=_EmbeddingAwareLoss())
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        emb = torch.randn(4, 8)
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': emb}))
        self.assertEqual(loss.shape, ())

    def test_extra_kwarg_ignored_when_not_in_sig(self):
        loss_fn = ConceptLoss(self.ann, binary=nn.BCEWithLogitsLoss())
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        # Should not raise even though BCEWithLogitsLoss doesn't take 'embeddings'
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': torch.randn(4, 8)}))
        self.assertEqual(loss.shape, ())

    def test_var_kwargs_term_receives_everything(self):
        loss_fn = ConceptLoss(self.ann, binary=_VarKwargsLoss())
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'extra_data': torch.ones(3)}))
        self.assertEqual(loss.shape, ())

    def test_composite_mixed_signatures(self):
        """Composite list: one term uses embeddings, the other doesn't."""
        loss_fn = ConceptLoss(
            self.ann,
            binary=[nn.BCEWithLogitsLoss(), _EmbeddingAwareLoss()],
            binary_weights=[1.0, 0.5],
        )
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        emb = torch.randn(4, 8)
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': emb}))
        self.assertEqual(loss.shape, ())
        # Verify embeddings affect the loss
        loss_zero = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': torch.zeros(4, 8)}))
        self.assertNotEqual(loss.item(), loss_zero.item())


class TestWeightedConceptLossKwargsForwarding(unittest.TestCase):
    """Test WeightedConceptLoss forwards extra kwargs to inner ConceptLoss."""

    def setUp(self):
        axis = Annotations(
            labels=('c1', 'c2', 't1'),
            cardinalities=[1, 1, 1],
            metadata={
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'},
                't1': {'type': 'discrete'},
            },
        )
        self.ann = axis

    def test_extra_kwargs_forwarded(self):
        loss_fn = WeightedConceptLoss(
            self.ann,
            concept_weight=0.5, task_weight=0.5,
            task_names=['t1'],
            binary=_EmbeddingAwareLoss(),
        )
        preds = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        emb = torch.randn(4, 8)
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': emb}))
        self.assertEqual(loss.shape, ())


class TestDepthWeightedKwargsForwarding(unittest.TestCase):
    """Test DepthWeightedConceptLoss forwards extra kwargs."""

    def setUp(self):
        from torch.distributions import Bernoulli
        from torch_concepts import ConceptGraph
        axis = Annotations(
            labels=['A', 'B'],
            cardinalities=[1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
            },
        )
        self.ann = axis
        adj = torch.tensor([[0., 1.], [0., 0.]])
        self.graph = ConceptGraph(adj, node_names=['A', 'B'])

    def test_extra_kwargs_forwarded(self):
        loss_fn = DepthWeightedConceptLoss(
            self.ann, self.graph,
            binary=_EmbeddingAwareLoss(),
        )
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        emb = torch.randn(4, 8)
        loss = loss_fn(ModelOutput(logits=preds, target=targets, extra={'embeddings': emb}))
        self.assertEqual(loss.shape, ())


# ======================================================================
# Categorical composite tests
# ======================================================================

class TestConceptLossCategoricalComposite(unittest.TestCase):
    """Test ConceptLoss with composite lists on categorical concepts."""

    def setUp(self):
        axis = Annotations(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={'cat1': {'type': 'discrete'}, 'cat2': {'type': 'discrete'}},
        )
        self.ann = axis

    def test_categorical_composite_forward(self):
        loss_fn = ConceptLoss(
            self.ann,
            categorical=[nn.CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
            categorical_weights=[1.0, 0.3],
        )
        preds = torch.randn(8, 8)  # 3 + 5
        targets = torch.cat([
            torch.randint(0, 3, (8, 1)),
            torch.randint(0, 5, (8, 1)),
        ], dim=1)
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_categorical_composite_gradient(self):
        loss_fn = ConceptLoss(
            self.ann,
            categorical=[nn.CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
        )
        preds = torch.randn(8, 8, requires_grad=True)
        targets = torch.cat([
            torch.randint(0, 3, (8, 1)),
            torch.randint(0, 5, (8, 1)),
        ], dim=1)
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        loss.backward()
        self.assertIsNotNone(preds.grad)


class TestMixedCompositeBothTypes(unittest.TestCase):
    """Test composite lists on BOTH binary and categorical simultaneously."""

    def test_mixed_composite(self):
        axis = Annotations(
            labels=('b1', 'cat1'),
            cardinalities=[1, 4],
            metadata={'b1': {'type': 'discrete'}, 'cat1': {'type': 'discrete'}},
        )
        ann = axis

        loss_fn = ConceptLoss(
            ann,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
            categorical=[nn.CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
            categorical_weights=[1.0, 0.3],
        )
        preds = torch.randn(8, 5)  # 1 binary + 4 categorical
        targets = torch.cat([
            torch.randint(0, 2, (8, 1)).float(),
            torch.randint(0, 4, (8, 1)),
        ], dim=1)
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)


# ======================================================================
# WeightedConceptLoss / DepthWeightedConceptLoss composite per-type
# ======================================================================

class TestWeightedConceptLossComposite(unittest.TestCase):
    """Test WeightedConceptLoss with per-type composite losses."""

    def test_composite_binary_with_regularizer(self):
        axis = Annotations(
            labels=('c1', 'c2', 't1'),
            cardinalities=[1, 1, 1],
            metadata={
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'},
                't1': {'type': 'discrete'},
            },
        )
        ann = axis

        loss_fn = WeightedConceptLoss(
            ann,
            concept_weight=0.7,
            task_weight=0.3,
            task_names=['t1'],
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
        )
        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)


class TestDepthWeightedConceptLossComposite(unittest.TestCase):
    """Test DepthWeightedConceptLoss with per-type composite losses."""

    def test_composite_binary_with_regularizer(self):
        from torch.distributions import Bernoulli
        from torch_concepts import ConceptGraph
        axis = Annotations(
            labels=['A', 'B', 'C'],
            cardinalities=[1, 1, 1],
            metadata={
                'A': {'type': 'discrete', 'distribution': Bernoulli},
                'B': {'type': 'discrete', 'distribution': Bernoulli},
                'C': {'type': 'discrete', 'distribution': Bernoulli},
            },
        )
        ann = axis
        adj = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])

        loss_fn = DepthWeightedConceptLoss(
            ann, graph,
            binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
        )
        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        loss = loss_fn(ModelOutput(logits=preds, target=targets))
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)


# ======================================================================
# _prepare_categorical tests
# ======================================================================

class TestPrepareCategorical(unittest.TestCase):
    """Test ConceptLoss._prepare_categorical helper."""

    def test_padding_and_shape(self):
        """Logits padded to max_card, stacked along batch dim."""
        axis = Annotations(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={'cat1': {'type': 'discrete'}, 'cat2': {'type': 'discrete'}},
        )
        ann = axis
        loss_fn = ConceptLoss(ann, categorical=nn.CrossEntropyLoss())

        preds = torch.randn(4, 8)  # 3 + 5
        targets = torch.cat([
            torch.randint(0, 3, (4, 1)),
            torch.randint(0, 5, (4, 1)),
        ], dim=1)

        cat_logits, cat_targets, cat_mask = loss_fn._prepare_categorical(preds, targets)
        # 2 concepts x 4 batch = 8 rows, padded to max_card=5 columns
        self.assertEqual(cat_logits.shape, (8, 5))
        self.assertEqual(cat_targets.shape, (8,))
        self.assertEqual(cat_mask.shape, (8, 5))
        # Padded positions for cat1 (card=3) should be -inf in columns 3,4
        self.assertTrue((cat_logits[:4, 3:] == float('-inf')).all())
        # Mask should be False at padded positions, True elsewhere
        self.assertTrue((cat_mask[:4, :3]).all())
        self.assertFalse((cat_mask[:4, 3:]).any())
        self.assertTrue((cat_mask[4:, :]).all())  # cat2 has max card, no padding

    def test_single_categorical_no_padding(self):
        axis = Annotations(
            labels=('cat1',),
            cardinalities=(4,),
            metadata={'cat1': {'type': 'discrete'}},
        )
        ann = axis
        loss_fn = ConceptLoss(ann, categorical=nn.CrossEntropyLoss())

        preds = torch.randn(6, 4)
        targets = torch.randint(0, 4, (6, 1))

        cat_logits, cat_targets, cat_mask = loss_fn._prepare_categorical(preds, targets)
        self.assertEqual(cat_logits.shape, (6, 4))
        self.assertEqual(cat_targets.shape, (6,))
        self.assertEqual(cat_mask.shape, (6, 4))
        # Single concept at max card — no padding, mask all True
        self.assertTrue(cat_mask.all())


if __name__ == '__main__':
    unittest.main()
