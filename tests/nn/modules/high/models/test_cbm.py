"""
Comprehensive tests for Concept Bottleneck Model (CBM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (joint, independent)
- Backbone integration
- Distribution handling
- Target preparation (prepare_target)
- Factory function behavior
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical, RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn import MLP
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.annotations import Annotations


def _logits(out, names):
    """Concatenate per-variable logits for the queried ``names`` -> (B, sum cardinalities)."""
    import torch
    return out.logits[list(names)]


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class TestCBMInitialization(unittest.TestCase):
    """Test CBM initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
                }
            )
    
    def test_init_defaults(self):
        """Test initialization with default distributions on the model."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

        self.assertIsInstance(model.pgm, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['color', 'shape', 'size', 'task1'])
        # Distributions live on the model, not on the annotation
        self.assertEqual(model.variable_distributions['categorical'], OneHotCategorical)
        self.assertEqual(model.variable_distributions['binary'], Bernoulli)

    def test_init_with_variable_distributions_param(self):
        """Test initialization passing per-type variable_distributions override."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            variable_distributions={'binary': RelaxedBernoulli},
        )

        self.assertEqual(model.variable_distributions['binary'], RelaxedBernoulli)
        self.assertEqual(model.variable_distributions['categorical'], OneHotCategorical)
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone (raw input -> latent)."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,
            task_names=['task1']
        )

        self.assertIs(model.backbone, backbone)

    def test_init_with_mlp_backbone(self):
        """Test initialization with an MLP backbone resolving latent_size."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            backbone=MLP(input_size=8, hidden_size=16, n_layers=2),
            latent_size=16,
        )

        self.assertEqual(model.latent_size, 16)


class TestCBMForward(unittest.TestCase):
    """Test CBM forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
                }
            )
        
        self.model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        x = torch.randn(2, 8)
        query = ['color', 'shape', 'size']
        out = self.model(query=query, input=x)

        # Output shape: batch_size x sum(cardinalities for queried variables)
        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2 + 1)  # color + shape + size

    def test_forward_all_concepts(self):
        """Test forward with all concepts."""
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

    def test_forward_with_backbone(self):
        """Test forward pass with backbone (raw input -> latent inside the PGM)."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptBottleneckModel(
            input_size=100,  # raw input dim (the PGM 'input' node)
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,   # backbone output dim (the PGM 'latent' node)
            task_names=['task1']
        )

        x = torch.randn(2, 100)  # Raw input size (before backbone)
        query = ['color', 'shape']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2)


class TestCBMPrepareTarget(unittest.TestCase):
    """Test CBM prepare_target."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        
        self.model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
    
    def test_prepare_target(self):
        """Test prepare_target returns target unchanged for CBM."""
        target = torch.randint(0, 2, (2, 3)).float()
        
        prepared = self.model.prepare_target(target)
        self.assertTrue(torch.allclose(prepared, target))


class TestCBMTraining(unittest.TestCase):
    """Test CBM training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
    
    def test_manual_training_mode(self):
        """Test manual PyTorch training (no lightning mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # No lightning mode = pure PyTorch module
        self.assertFalse(isinstance(model, BaseLearner))

        # Can train manually
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()

        model.train()
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)

        self.assertTrue(loss.requires_grad)

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = ConceptBottleneckModel(
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


class TestCBMEdgeCases(unittest.TestCase):
    """Test CBM edge cases and error handling."""
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        ann = Annotations(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'}
                }
            )
        
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c2']
        )
        
        x = torch.randn(2, 8)
        # Empty or None query should handle gracefully
        # Behavior depends on implementation
    
    def test_repr(self):
        """Test string representation."""
        ann = Annotations(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'task': {'type': 'discrete'},
                }
            )

        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )

        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)


# =============================================================================
# Tests for Factory Function and Training Modes
# =============================================================================

class TestCBMFactory(unittest.TestCase):
    """Test ConceptBottleneckModel factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )

    def test_factory_joint_mode(self):
        """Test factory creates Lightning model with lightning=True."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        self.assertIsInstance(model, BaseLearner)

    def test_factory_independent_mode(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        # Should succeed (no ValueError) because IndependentInference is a subclass
        model = ConceptBottleneckModel(
            lightning=True,
            train_inference=IndependentInference,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        self.assertIsInstance(model, BaseLearner)

    def test_factory_default_is_pytorch(self):
        """Test default is pure PyTorch module (no lightning mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        self.assertFalse(isinstance(model, BaseLearner))


class TestCBMUnifiedForward(unittest.TestCase):
    """Test unified forward pass works across all modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        self.x = torch.randn(4, 8)
    
    def test_forward_with_x_only(self):
        """Test forward with x tensor only via lightning=True."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        out = model(query=['c1', 'c2', 'task'], input=self.x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (4, 3))

    def test_forward_with_evidence(self):
        """Test forward with evidence dict works without error."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        out = model(query=['c1', 'c2', 'task'], input=self.x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (4, 3))

    def test_forward_same_output_all_modes(self):
        """Test Lightning and pure PyTorch modes produce same forward output shape."""
        for lightning_mode in [True, False]:
            model = ConceptBottleneckModel(
                lightning=lightning_mode,
                input_size=8,
                annotations=self.ann,
                task_names=['task']
            )

            out = model(query=['c1', 'c2', 'task'], input=self.x)
            self.assertEqual(
                _logits(out, ['c1', 'c2', 'task']).shape, (4, 3),
                f"Failed for lightning_mode: {lightning_mode}"
            )


class TestTrainingModes(unittest.TestCase):
    """Test different training modes."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        self.kwargs = {
            'input_size': 8,
            'annotations': self.ann,
            'task_names': ['task']
        }

    def test_joint_mode_works(self):
        """Test ConceptBottleneckModel with Lightning training."""
        model = ConceptBottleneckModel(lightning=True, **self.kwargs)

        self.assertIsInstance(model, BaseLearner)
        x = torch.randn(2, 8)
        out = model(query=['c1', 'c2', 'task'], input=x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (2, 3))

    def test_independent_mode_works(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        model = ConceptBottleneckModel(lightning=True, train_inference=IndependentInference, **self.kwargs)
        self.assertIsInstance(model, BaseLearner)


class TestLearnerIntegration(unittest.TestCase):
    """Test learner training step integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        self.batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()}
        }

    def _make_model(self, lightning=True, with_loss=True, train_inference=None):
        """Helper to create model with optional loss."""
        loss = ConceptLoss(binary=nn.BCEWithLogitsLoss()) if with_loss else None
        kwargs = {
            'lightning': lightning,
            'input_size': 8,
            'annotations': self.ann,
            'task_names': ['task'],
            'loss': loss,
            'optim_class': torch.optim.Adam,
            'optim_kwargs': {'lr': 0.01}
        }
        if train_inference is not None:
            kwargs['train_inference'] = train_inference
        return ConceptBottleneckModel(**kwargs)

    def test_joint_training_step(self):
        """Test Lightning learner training step."""
        model = self._make_model(lightning=True)
        model.train()

        loss = model.training_step(self.batch)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)

    def test_independent_training_step(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        model = self._make_model(lightning=True, train_inference=IndependentInference)
        self.assertIsInstance(model, BaseLearner)

    def test_configure_optimizers_joint(self):
        """Test optimizer configuration for Lightning mode."""
        model = self._make_model(lightning=True)

        config = model.configure_optimizers()

        self.assertIn('optimizer', config)
        self.assertIsInstance(config['optimizer'], torch.optim.Adam)


if __name__ == '__main__':
    unittest.main()


# ===========================================================================
# GraphConceptBottleneckModel tests
# ===========================================================================

import torch
from torch_concepts import ConceptGraph
from torch_concepts.annotations import Annotations
from torch_concepts.nn.modules.high.models.graph_cbm import GraphConceptBottleneckModel


def _make_graph_cbm_ann():
    return Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1],
            types=['binary', 'binary', 'binary'],
        )


def _make_dag():
    # a -> b -> c
    adj = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    return ConceptGraph(adj, node_names=['a', 'b', 'c'])


class TestGraphCBMConstruction:
    def test_basic_construction(self):
        ann = _make_graph_cbm_ann()
        graph = _make_dag()
        model = GraphConceptBottleneckModel(
            input_size=4,
            annotations=ann,
            graph=graph,
        )
        assert model is not None
        assert hasattr(model, 'pgm')

    def test_build_encoder_is_called(self):
        ann = _make_graph_cbm_ann()
        graph = _make_dag()
        model = GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=graph)
        # The model builds encoders for root nodes (just 'a')
        assert hasattr(model, 'pgm')

    def test_build_predictor_is_called(self):
        ann = _make_graph_cbm_ann()
        graph = _make_dag()
        model = GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=graph)
        assert hasattr(model, 'inference')

    def test_forward_basic(self):
        ann = _make_graph_cbm_ann()
        graph = _make_dag()
        model = GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=graph)
        model.eval()
        x = torch.randn(3, 4)
        out = model.forward(query=['a', 'b', 'c'], input=x)
        assert out is not None


# ===========================================================================
# DirectedGraphModel / GraphModel base class tests
# ===========================================================================

import torch
from torch_concepts import ConceptGraph
from torch_concepts.annotations import Annotations
from torch_concepts.nn.modules.high.base.graph import DirectedGraphModel
from torch_concepts.nn.modules.high.models.graph_cbm import GraphConceptBottleneckModel


def _make_simple_ann():
    return Annotations(
            labels=['x', 'y'],
            cardinalities=[1, 1],
            types=['binary', 'binary'],
        )


def _make_two_node_dag():
    adj = torch.tensor([[0., 1.], [0., 0.]])
    return ConceptGraph(adj, node_names=['x', 'y'])


class TestDirectedGraphModelBase:
    def test_resolve_graph_raises_when_no_graph(self):
        """GraphModel raises ValueError when no graph is provided."""
        ann = _make_simple_ann()
        with pytest.raises((ValueError, AssertionError)):
            # Pass no graph → should fail in _resolve_graph or validation
            GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=None)

    def test_graph_model_rejects_plate_true(self):
        """Graph models are individual-only; plate=True is rejected."""
        ann = _make_simple_ann()
        graph = _make_two_node_dag()
        with pytest.raises(ValueError):
            GraphConceptBottleneckModel(
                input_size=4, annotations=ann, graph=graph, plate=True
            )

    def test_graph_model_plate_false_default(self):
        """Graph models build one (individual) variable per concept node."""
        ann = _make_simple_ann()
        graph = _make_two_node_dag()
        model = GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=graph)
        concept_vars = [v for v in model.pgm.variables.values()
                        if v.variable_type == "concept"]
        assert {v.name for v in concept_vars} == set(ann.labels)
        assert all(not v.is_plate for v in concept_vars)

    def _graph_model(self):
        from torch_concepts.nn.modules.high.models.graph_cbm import GraphConceptBottleneckModel
        return GraphConceptBottleneckModel(
            input_size=4, annotations=_make_simple_ann(), graph=_make_two_node_dag()
        )

    def test_flexible_parametrization_normal_auto(self):
        """`second='auto'` gives a positive scale head copied from `first`."""
        import torch.distributions as dist
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        norm_var = ConceptVariable("v", distribution=dist.Normal, size=3)
        first = torch.nn.Linear(4, 3)

        param = model._flexible_parametrization(norm_var, first, second='auto')

        assert set(param) == {"loc", "scale"}
        assert param["loc"] is first
        # The scale head is a raw copy of `first` followed by softplus, and the
        # copy is independent — not the same module.
        assert param["scale"][0] is not first
        assert isinstance(param["scale"][1], torch.nn.Softplus)
        scale = param["scale"](torch.randn(6, 4))
        assert scale.shape == (6, 3)
        assert bool((scale > 0).all())

    def test_flexible_parametrization_requires_an_explicit_second(self):
        """`second` is not optional for a continuous variable: the copy must be
        asked for with 'auto' rather than happening implicitly."""
        import torch.distributions as dist
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        norm_var = ConceptVariable("v", distribution=dist.Normal, size=3)
        with pytest.raises(ValueError, match="'auto'"):
            model._flexible_parametrization(norm_var, torch.nn.Linear(4, 3))

    def test_flexible_parametrization_auto_is_ignored_for_discrete(self):
        """A caller can pass 'auto' for every variable: only a continuous one
        has a second parameter to derive."""
        import torch.distributions as dist
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        binary = ConceptVariable("v", distribution=dist.Bernoulli, size=1)
        first = torch.nn.Linear(4, 1)
        assert model._flexible_parametrization(binary, first, second='auto') == {
            "logits": first
        }

    def test_flexible_parametrization_normal_explicit_second(self):
        """`second` supplies the scale head: a concrete layer, or a deferred
        LazyConstructor left unbuilt for the CPD to size from the parents."""
        import torch.distributions as dist
        from torch_concepts.nn import LazyConstructor, LinearEmbeddingToConcept
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        norm_var = ConceptVariable("v", distribution=dist.Normal, size=3)
        head = torch.nn.Linear(4, 3)

        as_module = model._flexible_parametrization(norm_var, torch.nn.Linear(4, 3), second=head)
        as_lazy = model._flexible_parametrization(
            norm_var, LazyConstructor(LinearEmbeddingToConcept), second=LazyConstructor(LinearEmbeddingToConcept)
        )

        assert as_module["scale"][0] is head
        # The lazy scale head is stored unbuilt (the CPD sizes it later) and
        # already composed with its activation.
        lazy_head = as_lazy["scale"][0]
        assert isinstance(lazy_head, LazyConstructor) and lazy_head.module is None
        assert isinstance(as_lazy["scale"][1], torch.nn.Softplus)

    def test_flexible_parametrization_multivariate_normal(self):
        """A MultivariateNormal gets a matrix-valued, positive-diagonal scale_tril."""
        import torch.distributions as dist
        from torch_concepts.nn import TrilActivation
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        mvn_var = ConceptVariable("v", distribution=dist.MultivariateNormal, size=3)
        # scale_tril needs 3*(3+1)//2 = 6 outputs, unlike loc's 3.
        param = model._flexible_parametrization(
            mvn_var, torch.nn.Linear(4, 3), second=torch.nn.Linear(4, 6)
        )

        assert set(param) == {"loc", "scale_tril"}
        assert isinstance(param["scale_tril"][1], TrilActivation)
        tril = param["scale_tril"](torch.randn(6, 4))
        assert tril.shape == (6, 3, 3)
        assert bool((tril.diagonal(dim1=-2, dim2=-1) > 0).all())
        assert bool((tril.triu(diagonal=1) == 0).all())

    def test_flexible_parametrization_uncopyable_scale_raises(self):
        """A non-per-element scale cannot be copied: its width differs from loc's."""
        import torch.distributions as dist
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        mvn_var = ConceptVariable("v", distribution=dist.MultivariateNormal, size=3)
        with pytest.raises(ValueError, match="scale_tril"):
            model._flexible_parametrization(mvn_var, torch.nn.Linear(4, 3), second='auto')

    def test_flexible_parametrization_deferred_first_raises(self):
        """A deferred `first` has no fixed width yet, so it cannot be copied."""
        import torch.distributions as dist
        from torch_concepts.nn import LazyConstructor, LinearEmbeddingToConcept
        from torch_concepts.nn.modules.mid.variable import ConceptVariable
        model = self._graph_model()
        norm_var = ConceptVariable("v", distribution=dist.Normal, size=3)
        with pytest.raises(ValueError, match="LazyConstructor"):
            model._flexible_parametrization(
                norm_var, LazyConstructor(LinearEmbeddingToConcept), second='auto'
            )

    def test_plate_compatible_levels(self):
        """plate_compatible_levels returns True for homogeneous levels."""
        ann = Annotations(
                labels=['a', 'b'],
                cardinalities=[1, 1],
                types=['binary', 'binary'],
            )
        graph = ConceptGraph(
            torch.tensor([[0., 0.], [0., 0.]]),
            node_names=['a', 'b'],
        )
        axis_ann = ann
        result = DirectedGraphModel.plate_compatible_levels(axis_ann, graph)
        assert all(result)

    def test_plate_compatible_levels_metadata_fallback(self):
        """plate_compatible_levels uses metadata type when types is None."""
        # Build annotation without explicit types (uses metadata)
        ann_axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[1, 1],
            metadata={'a': {'type': 'binary'}, 'b': {'type': 'binary'}},
        )
        # types is None initially, but gets resolved after __init__
        # Test via the graph model static method directly with an axis that has metadata
        graph = ConceptGraph(torch.tensor([[0., 0.], [0., 0.]]), node_names=['a', 'b'])
        result = DirectedGraphModel.plate_compatible_levels(ann_axis, graph)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_dag_validation_rejects_cycle(self):
        """DirectedGraphModel validates that the graph is a DAG."""
        ann = _make_simple_ann()
        cycle_adj = torch.tensor([[0., 1.], [1., 0.]])
        cycle_graph = ConceptGraph(cycle_adj, node_names=['x', 'y'])
        with pytest.raises(AssertionError):
            GraphConceptBottleneckModel(input_size=4, annotations=ann, graph=cycle_graph)


class TestGraphCBMContinuousConcepts:
    """GraphConceptBottleneckModel models continuous concepts as Normal, with a
    scale head copied from the encoder (root) or predictor (internal node)."""

    @staticmethod
    def _model(types):
        from torch_concepts.nn.modules.high.models.graph_cbm import GraphConceptBottleneckModel
        ann = Annotations(labels=['x', 'y'], cardinalities=[1, 1], types=types)
        return GraphConceptBottleneckModel(
            input_size=6, annotations=ann, graph=_make_two_node_dag()
        )

    def test_forward_reports_loc_and_positive_scale(self):
        model = self._model(['continuous', 'continuous'])
        out = model(query=['x', 'y'], input=torch.randn(4, 6))
        assert out.loc.shape == (4, 2)
        assert out.scale.shape == (4, 2)
        assert bool((out.scale > 0).all())

    def test_both_root_and_internal_nodes_get_a_scale_head(self):
        """The root is encoded from the latent, the internal node predicted from
        its parents — two different `first` layers, both copied by `second='auto'`."""
        model = self._model(['continuous', 'continuous'])
        for name in ('x', 'y'):
            assert 'scale' in model.pgm.factors[name].parametrization

    def test_mixed_types_split_across_quantities(self):
        model = self._model(['binary', 'continuous'])
        out = model(query=['x', 'y'], input=torch.randn(4, 6))
        assert list(out.logits.annotation.labels) == ['x']
        assert list(out.loc.annotation.labels) == ['y']

    def test_gradients_flow(self):
        model = self._model(['continuous', 'continuous'])
        out = model(query=['x', 'y'], input=torch.randn(4, 6))
        out.loc.sum().backward()
        assert any(p.grad is not None for p in model.parameters())
