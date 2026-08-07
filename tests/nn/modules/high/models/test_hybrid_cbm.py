"""
Comprehensive tests for the Hybrid Concept Bottleneck Model (HybridCBM).

A HybridCBM extends a standard CBM bottleneck with a set of *unsupervised*
latent dimensions. These tests cover the behaviour that is specific to that
extension (everything shared with the plain CBM is already covered by
``test_cbm.py``):

- Initialization and validation of ``additional_dims`` / ``additional_dim_types``
- The unsupervised dimensions entering the PGM as non-interpretable
  ``EmbeddingVariable`` nodes (never supervised / teacher-forced)
- Plate grouping of the unsupervised dimensions (a single plate when
  homogeneous, split per type otherwise, one variable each with ``plate=False``)
- The per-type distribution policy for the unsupervised dimensions
  (continuous -> Delta, binary -> Bernoulli)
- Forward pass / output shapes, including querying the unsupervised dimensions
- Gradient flow into the unsupervised encoders through the task loss
- Reduction to a plain CBM when ``additional_dims == 0``
- Collision-free naming of the unsupervised dimensions
- A regression test that the train/eval inference engines share the model's
  ``pgm`` (so ``model.parameters()`` actually drives the forward pass)
"""
import pytest
import unittest

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts.annotations import Annotations
from torch_concepts.distributions import Delta
from torch_concepts.nn import MLP
from torch_concepts.nn.modules.high.models.hybrid_cbm import (
    HybridConceptBottleneckModel,
)
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.mid.variable import (
    ConceptVariable,
    EmbeddingVariable,
)


def _logits(out, names):
    """Concatenate the queried concepts' logits into a ``(B, sum(card))`` tensor."""
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)


def _binary_ann(concepts=('c1', 'c2', 'c3'), task='task'):
    """Homogeneous binary annotation: ``concepts`` + a single binary ``task``."""
    labels = list(concepts) + [task]
    return Annotations(
        labels=labels,
        cardinalities=[1] * len(labels),
        types=['binary'] * len(labels),
    )


def _task_head(model, task_name):
    """The ``LinearConceptToConcept`` producing ``task_name``'s logits.

    Works for both building layouts: the plate layout stores the task CPD under
    the plate name ``"tasks"``, the individual layout under the task's own name.
    """
    factors = model.pgm.factors
    cpd = factors['tasks'] if 'tasks' in factors else factors[task_name]
    return cpd.parametrization['logits']


class TestHybridCBMInitialization(unittest.TestCase):
    """Construction and the derived unsupervised-dimension bookkeeping."""

    def setUp(self):
        self.ann = _binary_ann()

    def test_basic_init(self):
        model = HybridConceptBottleneckModel(
            input_size=6,
            annotations=self.ann,
            additional_dims=4,
            task_names=['task'],
        )
        self.assertIsInstance(model.pgm, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.additional_dims, 4)
        self.assertEqual(len(model.unsup_names), 4)

    def test_default_is_pure_pytorch(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'],
        )
        self.assertFalse(isinstance(model, BaseLearner))

    def test_lightning_mode(self):
        model = HybridConceptBottleneckModel(
            lightning=True, input_size=6, annotations=self.ann,
            additional_dims=2, task_names=['task'],
        )
        self.assertIsInstance(model, BaseLearner)

    def test_unsup_names_are_unique_and_extra(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=3,
            task_names=['task'],
        )
        # Unsupervised names are distinct from the annotated concepts...
        self.assertEqual(len(set(model.unsup_names)), 3)
        self.assertFalse(set(model.unsup_names) & set(self.ann.labels))
        # ...and the supervised set is exactly the annotated (non-task) concepts.
        self.assertEqual(model.supervised_concept_names, ['c1', 'c2', 'c3'])

    def test_backbone_and_latent_size(self):
        model = HybridConceptBottleneckModel(
            input_size=20,
            annotations=self.ann,
            additional_dims=2,
            task_names=['task'],
            backbone=MLP(input_size=20, hidden_size=16, n_layers=1),
            latent_size=16,
        )
        self.assertEqual(model.latent_size, 16)

    def test_additional_dim_types_string_broadcast(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=3,
            task_names=['task'], additional_dim_types='binary',
        )
        # A single string is broadcast to every additional dimension.
        for name in model.unsup_names:
            self.assertEqual(model.concept_annotations.concept(name).type, 'binary')


class TestHybridCBMStructure(unittest.TestCase):
    """The assembled PGM: variable kinds, plate grouping, task head width."""

    def setUp(self):
        self.ann = _binary_ann()

    def test_homogeneous_unsup_is_a_single_plate(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )
        # Homogeneous concepts + homogeneous continuous unsup dims -> one plate
        # each, so the PGM has exactly these variables.
        self.assertEqual(
            set(model.pgm.variables),
            {'input', 'latent', 'concepts', model.unsup_plate_name, 'tasks'},
        )

    def test_unsup_dims_are_embedding_variables(self):
        """Unsupervised dims must be non-interpretable (embedding) variables."""
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )
        variables = model.pgm.variables
        self.assertIsInstance(variables[model.unsup_plate_name], EmbeddingVariable)
        self.assertEqual(variables[model.unsup_plate_name].variable_type, 'embedding')
        # The supervised concepts and tasks stay interpretable concept variables.
        self.assertIsInstance(variables['concepts'], ConceptVariable)
        self.assertIsInstance(variables['tasks'], ConceptVariable)

    def test_mixed_unsup_types_split_into_per_type_plates(self):
        """A heterogeneous unsup group splits into one plate per type."""
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'],
            additional_dim_types=['continuous', 'binary'],
        )
        # Two homogeneous unsup plates (continuous + binary), each an embedding.
        unsup_vars = [
            v for v in model.pgm.variables.values()
            if v.variable_type == 'embedding' and v.name not in ('input', 'latent')
        ]
        self.assertEqual(len(unsup_vars), 2)
        self.assertEqual(
            {v.distribution for v in unsup_vars}, {Delta, Bernoulli},
        )

    def test_plate_false_gives_individual_variables(self):
        """``plate=False`` builds one variable per concept and per unsup dim."""
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'], plate=False,
        )
        # One variable per unsupervised dimension, each an EmbeddingVariable.
        for name in model.unsup_names:
            self.assertIn(name, model.pgm.variables)
            self.assertEqual(model.pgm.variables[name].variable_type, 'embedding')
        # And one variable per supervised concept.
        for name in model.supervised_concept_names:
            self.assertIn(name, model.pgm.variables)

    def test_task_head_consumes_concepts_and_unsup_dims(self):
        """The task predictor's input width == supervised concepts + unsup dims."""
        for dim_types in ('continuous',
                          ['continuous', 'binary', 'continuous']):
            model = HybridConceptBottleneckModel(
                input_size=6, annotations=self.ann, additional_dims=3,
                task_names=['task'], additional_dim_types=dim_types,
            )
            head = _task_head(model, 'task')
            # 3 supervised binary concepts + 3 unsupervised dims.
            self.assertEqual(
                head.predictor.in_features, 3 + 3,
                f"wrong task-head width for dim_types={dim_types}",
            )


class TestHybridCBMUnsupervisedDistributions(unittest.TestCase):
    """Per-type distribution policy for the unsupervised dimensions."""

    def setUp(self):
        self.ann = _binary_ann()

    def test_continuous_dims_use_delta(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'], additional_dim_types='continuous',
        )
        self.assertIs(model.pgm.variables[model.unsup_plate_name].distribution, Delta)

    def test_binary_dims_use_bernoulli(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'], additional_dim_types='binary',
        )
        self.assertIs(model.pgm.variables[model.unsup_plate_name].distribution, Bernoulli)


class TestHybridCBMForward(unittest.TestCase):
    """Forward pass and output shapes."""

    def setUp(self):
        self.ann = _binary_ann()
        self.model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )

    def test_forward_concepts_and_task(self):
        x = torch.randn(8, 6)
        query = ['c1', 'c2', 'c3', 'task']
        out = self.model(query=query, input=x)
        logits = _logits(out, query)
        self.assertEqual(logits.shape, (8, 4))

    def test_forward_query_continuous_unsup_dims(self):
        """Continuous unsupervised dims are deterministic -> a ``value`` param."""
        x = torch.randn(8, 6)
        out = self.model(query=self.model.unsup_names, input=x)
        for name in self.model.unsup_names:
            self.assertIn('value', out.params[name])
            self.assertEqual(out.params[name]['value'].shape, (8, 1))

    def test_forward_query_binary_unsup_dims(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'], additional_dim_types='binary',
        )
        x = torch.randn(5, 6)
        out = model(query=model.unsup_names, input=x)
        for name in model.unsup_names:
            self.assertIn('logits', out.params[name])
            self.assertEqual(out.params[name]['logits'].shape, (5, 1))

    def test_fully_observed_query_excludes_unsup_dims(self):
        """Teacher-forcing targets cover only the supervised concept variables."""
        gt = torch.randint(0, 2, (8, 4))  # c1, c2, c3, task
        query = self.model.fully_observed_query(gt)
        # Plate layout: one entry per concept variable, none for the unsup plate.
        self.assertEqual(set(query), {'concepts', 'tasks'})
        self.assertEqual(query['concepts'].shape, (8, 3))
        self.assertEqual(query['tasks'].shape, (8, 1))


class TestHybridCBMTraining(unittest.TestCase):
    """Gradient flow, parameter updates, and inference-engine wiring."""

    def setUp(self):
        self.ann = _binary_ann()

    def test_gradients_flow_into_unsup_encoder(self):
        """The task loss backpropagates into the unsupervised encoders.

        The unsupervised dimensions are never in the loss directly, but they
        feed the task predictor, so a task loss must still reach their encoder.
        """
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )
        model.train()
        x = torch.randn(8, 6)
        out = model(query=['c1', 'c2', 'c3', 'task'], input=x)
        loss = _logits(out, ['c1', 'c2', 'c3', 'task']).sum()
        loss.backward()

        # continuous unsup dims -> Delta -> 'value' parametrization.
        unsup_cpd = model.pgm.factors[model.unsup_plate_name]
        grad = unsup_cpd.parametrization['value'].encoder.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_parameters_update(self):
        """A few (SGD) optimizer steps actually move the model's parameters."""
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )
        model.train()
        # SGD (not Adam) keeps this runnable on CPU-only boxes whose torch build
        # trips an accelerator health-check inside the fused Adam step.
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss_fn = nn.BCEWithLogitsLoss()

        x = torch.randn(16, 6)
        target = torch.randint(0, 2, (16, 4)).float()
        query = ['c1', 'c2', 'c3', 'task']

        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        first = None
        for _ in range(25):
            opt.zero_grad()
            out = model(query=query, input=x)
            loss = loss_fn(_logits(out, query), target)
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()

        self.assertLess(loss.item(), first)
        changed = [
            n for n, p in model.named_parameters()
            if not torch.allclose(p.detach(), before[n])
        ]
        self.assertTrue(changed, "no parameters changed during training")

    def test_inference_engines_share_pgm(self):
        """Regression: train/eval engines wrap the *same* pgm as the module.

        If the engines wrapped a stale pgm, ``model.parameters()`` would not
        include the parameters actually used in the forward pass, so training
        would silently optimise nothing.
        """
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=4,
            task_names=['task'],
        )
        self.assertIs(model.eval_inference.pgm, model.pgm)
        self.assertIs(model.train_inference.pgm, model.pgm)

        pgm_param_ids = {id(p) for p in model.pgm.parameters()}
        model_param_ids = {id(p) for p in model.parameters()}
        self.assertTrue(pgm_param_ids)
        self.assertTrue(pgm_param_ids <= model_param_ids)


class TestHybridCBMReducesToCBM(unittest.TestCase):
    """With no additional dimensions the model is a plain CBM."""

    def setUp(self):
        self.ann = _binary_ann(concepts=('c1', 'c2'))

    def test_zero_additional_dims_structure(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=0,
            task_names=['task'],
        )
        self.assertEqual(model.unsup_names, [])
        self.assertEqual(
            set(model.pgm.variables), {'input', 'latent', 'concepts', 'tasks'},
        )

    def test_zero_additional_dims_matches_cbm_shapes(self):
        hybrid = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=0,
            task_names=['task'],
        )
        cbm = ConceptBottleneckModel(
            input_size=6, annotations=self.ann, task_names=['task'],
        )
        x = torch.randn(4, 6)
        query = ['c1', 'c2', 'task']
        self.assertEqual(
            _logits(hybrid(query=query, input=x), query).shape,
            _logits(cbm(query=query, input=x), query).shape,
        )

    def test_negative_additional_dims_clamped_to_zero(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=-3,
            task_names=['task'],
        )
        self.assertEqual(model.additional_dims, 0)
        self.assertEqual(model.unsup_names, [])


class TestHybridCBMNaming(unittest.TestCase):
    """Collision-free naming of the unsupervised dimensions."""

    def test_name_clash_escalates_prefix(self):
        # An annotation that already uses the default unsupervised prefix.
        ann = Annotations(
            labels=['__unsup_0', 'c2', 'task'],
            cardinalities=[1, 1, 1],
            types=['binary', 'binary', 'binary'],
        )
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=ann, additional_dims=1, task_names=['task'],
        )
        # The prefix gains an underscore to avoid clashing with '__unsup_0'.
        self.assertEqual(model.unsup_names, ['___unsup_0'])
        self.assertNotIn(model.unsup_plate_name, ann.labels)
        # The pre-existing '__unsup_0' remains a normal supervised concept.
        self.assertIn('__unsup_0', model.supervised_concept_names)


class TestHybridCBMValidation(unittest.TestCase):
    """Validation of the additional-dimension arguments."""

    def setUp(self):
        self.ann = _binary_ann()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            HybridConceptBottleneckModel(
                input_size=6, annotations=self.ann, additional_dims=3,
                task_names=['task'], additional_dim_types=['continuous', 'binary'],
            )

    def test_categorical_unsup_dims_rejected(self):
        with pytest.raises(ValueError):
            HybridConceptBottleneckModel(
                input_size=6, annotations=self.ann, additional_dims=2,
                task_names=['task'], additional_dim_types='categorical',
            )

    def test_invalid_additional_dim_types_type_raises(self):
        with pytest.raises(ValueError):
            HybridConceptBottleneckModel(
                input_size=6, annotations=self.ann, additional_dims=2,
                task_names=['task'], additional_dim_types=5,
            )

    def test_repr(self):
        model = HybridConceptBottleneckModel(
            input_size=6, annotations=self.ann, additional_dims=2,
            task_names=['task'],
        )
        self.assertIsInstance(repr(model), str)


if __name__ == '__main__':
    unittest.main()
