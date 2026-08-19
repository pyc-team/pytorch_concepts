"""
Set of tests for the Probabilistic Concept Bottleneck Model (ProbCBM).

Everything shared with the plain CBM is already covered by ``test_cbm.py``, so
these tests cover only what is specific to representing each concept as a
Gaussian embedding decoded from anchor distances:
- Construction and the binary-concept restriction
- The PGM structure: one ``Normal`` embedding variable and one anchor-decoded
  concept variable per plate group, plus the shared anchor/projection tables
- Both building layouts (a single plate vs one variable per concept)
- Forward pass / output shapes, including categorical tasks
- Fidelity to the paper's equations, checked numerically against a literal
  transcription of each one (concept logits, class logits, VIB, uncertainty)
- The VIB regulariser and per-concept uncertainty helpers
- Deterministic (mean) vs ancestral (sampled) inference
- Both task heads: the paper's, which reads the concept embeddings, and the
  opt-in ``use_anchor_interpolation`` one, which reads the activations
- Concept interventions: that swapping a predicted embedding for its
  ground-truth anchor is what the model does
- Gradient flow / parameter updates
- The ``lightning=True`` training recipe: the sequential stages and what each
  freezes, the backbone warm-up, when ``p_replace`` is on, and the two
  learning rates
"""
import copy
import pytest
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.annotations import Annotations
from torch_concepts.data import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    AncestralSamplingInference,
    AnchorPredictor,
    DeterministicInference,
    MLP,
    ProbCBM,
)
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.nn.modules.low.sequential import Sequential
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.mid.variable import (
    ConceptVariable,
    EmbeddingVariable,
)


def _binary_ann(concepts=('c1', 'c2', 'c3'), task='task'):
    """
    Homogeneous binary annotation: ``concepts`` + a single binary ``task``.
    """
    labels = list(concepts) + [task]
    return Annotations(
        labels=labels,
        cardinalities=[1] * len(labels),
        types=['binary'] * len(labels),
    )


def _model(ann=None, **kwargs):
    """
    A small ProbCBM over ``ann`` (the homogeneous binary annotation by default).
    """
    ann = ann or _binary_ann()
    return ProbCBM(
        input_size=6,
        annotations=ann,
        task_names=['task'],
        embedding_size=8,
        class_embedding_size=16,
        **kwargs,
    )


class _StochasticLatentProbCBM(ProbCBM):
    """
    A ProbCBM whose ``latent`` is itself a ``Normal`` embedding rather than a
    deterministic one, used to check that only the concept embeddings are
    treated as the model's probabilistic ones.
    """

    def _input_latent_block(self):
        input_var = EmbeddingVariable(
            "input",
            distribution=Delta,
            shape=self.input_size,
        )
        latent_var = EmbeddingVariable(
            "latent",
            distribution=Normal,
            size=self.latent_size,
        )
        return (
            input_var,
            latent_var,
            ParametricCPD(
                input_var,
                parents=[],
                parametrization=LearnablePrior(input_var.shape),
            ),
            ParametricCPD(
                latent_var,
                parents=[input_var],
                parametrization={
                    "loc": self.backbone,
                    "scale": Sequential(
                        copy.deepcopy(self.backbone),
                        nn.Softplus(),
                    ),
                },
            ),
        )


class TestProbCBMInit(unittest.TestCase):
    """
    Construction and validation of the concept types.
    """

    def test_basic_init(self):
        model = _model()
        self.assertIsInstance(model.pgm, nn.Module)
        self.assertFalse(isinstance(model, BaseLearner))
        # The class head interpolates the very tables the concepts were
        # decoded against, so an intervention substitutes exactly the anchors
        # the concept probabilities are measured against.
        head = model.pgm.factors['tasks'].parametrization['logits']
        encoder = model.pgm.factors['concepts'].parametrization['logits']
        self.assertIs(head.source_anchors[0], encoder.anchors)
        self.assertIs(encoder, model.concept_predictors[0])
        self.assertIs(head.projection, model.class_projection)
        self.assertEqual(encoder.anchors.n_items, 3)

    def test_lightning_mode(self):
        self.assertIsInstance(_model(lightning=True), BaseLearner)

    def test_rejects_non_binary_concepts(self):
        ann = Annotations(
            labels=['c1', 'c2', 'task'],
            cardinalities=[3, 1, 1],
            types=['categorical', 'binary', 'binary'],
        )
        with pytest.raises(ValueError):
            ProbCBM(
                input_size=6,
                annotations=ann,
                task_names=['task'],
            )

    def test_categorical_task_is_allowed(self):
        """
        Only the *non-task* concepts need to be binary.
        """
        ann = Annotations(
            labels=['c1', 'c2', 't'],
            cardinalities=[1, 1, 3],
            types=['binary', 'binary', 'categorical'],
        )
        model = ProbCBM(
            input_size=6,
            annotations=ann,
            task_names=['t'],
            embedding_size=8,
            class_embedding_size=16,
        )
        out = model(query=['t'], input=torch.randn(4, 6))
        self.assertEqual(out.params['t']['logits'].shape, (4, 3))


class TestProbCBMStructure(unittest.TestCase):
    """
    The assembled PGM: variable kinds, plate grouping, embedding shapes.
    """

    def test_plate_layout_variables(self):
        model = _model()
        variables = model.pgm.variables
        # One batched Normal embedding + one concept plate + one task plate
        self.assertEqual(
            set(variables),
            {'input', 'latent', 'concepts__emb', 'concepts', 'tasks'},
        )
        self.assertIsInstance(variables['concepts__emb'], EmbeddingVariable)
        self.assertIs(variables['concepts__emb'].distribution, Normal)
        self.assertIsInstance(variables['concepts'], ConceptVariable)
        self.assertIs(variables['concepts'].distribution, Bernoulli)
        # The embedding event stacks (n_concepts, embedding_size)
        self.assertEqual(tuple(variables['concepts__emb'].shape), (3, 8))

    def test_individual_layout_variables(self):
        """
        ``plate=False`` builds one embedding and one concept per concept.
        """
        model = _model(plate=False)
        emb = [
            v for v in model.pgm.variables.values()
            if (
                (v.variable_type == 'embedding') and
                (v.name not in ('input', 'latent'))
            )
        ]
        self.assertEqual(
            {v.name for v in emb},
            {'c1__emb', 'c2__emb', 'c3__emb'},
        )
        for v in emb:
            self.assertEqual(tuple(v.shape), (8,))

    def test_embedding_query_names(self):
        self.assertEqual(_model().embedding_query_names, ['concepts__emb'])
        self.assertEqual(
            _model(plate=False).embedding_query_names,
            ['c1__emb', 'c2__emb', 'c3__emb'],
        )

    def test_embedding_query_names_ignore_other_normal_embeddings(self):
        """
        Only the *concept* embeddings count, whatever else the graph holds.

        A subclass is free to add other ``Normal`` embedding variables — a
        variational ``latent`` being the obvious one — and those must not be
        swept up, or the uncertainty and VIB helpers would silently reinterpret
        their width as extra concepts.
        """
        model = _StochasticLatentProbCBM(
            input_size=6,
            annotations=_binary_ann(),
            task_names=['task'],
            embedding_size=8,
            class_embedding_size=16,
            backbone=MLP(input_size=6, hidden_size=24, n_layers=1),
            latent_size=24,  # divisible by embedding_size, so a stray
        )                    # capture would reshape cleanly and go unnoticed
        model.eval()
        self.assertEqual(model.embedding_query_names, ['concepts__emb'])
        with torch.no_grad():
            out = model(query=model.embedding_query_names, input=torch.randn(4, 6))
        self.assertEqual(model.concept_uncertainty(out).shape, (4, 3))

    def test_task_head_reads_the_embeddings_by_default(self):
        """
        The paper's task head is a function of the concept embeddings, so the
        task CPD hangs off the embedding variables rather than the concepts.
        """
        model = _model()
        parents = {p.name for p in model.pgm.factors['tasks'].parents}
        self.assertEqual(parents, {'concepts__emb'})
        self.assertIsInstance(
            model.pgm.factors['tasks'].parametrization['logits'],
            AnchorPredictor,
        )

    def test_task_head_reads_the_concepts_when_interpolating(self):
        """
        ``use_anchor_interpolation`` puts the task head behind the bottleneck.
        """
        model = _model(use_anchor_interpolation=True)
        parents = {p.name for p in model.pgm.factors['tasks'].parents}
        self.assertEqual(parents, {'concepts'})
        self.assertIsInstance(
            model.pgm.factors['tasks'].parametrization['logits'],
            AnchorPredictor,
        )


class TestProbCBMForward(unittest.TestCase):
    """
    Forward pass, output shapes, and the uncertainty helpers.
    """

    def setUp(self):
        self.model = _model()
        self.x = torch.randn(8, 6)

    def test_forward_concepts_and_task(self):
        out = self.model(query=['c1', 'c2', 'c3', 'task'], input=self.x)
        self.assertEqual(out.logits[['c1', 'c2', 'c3', 'task']].shape, (8, 4))

    def test_embedding_params_available(self):
        out = self.model(
            query=['c1', 'c2', 'c3'] + self.model.embedding_query_names,
            input=self.x,
        )
        params = out.params['concepts__emb']
        self.assertIn('loc', params)
        self.assertIn('scale', params)
        # The location head normalises each concept embedding onto the sphere
        loc = params['loc'].reshape(8, 3, 8)
        norms = loc.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        )
        # And the scale is strictly positive (Softplus)
        self.assertTrue(bool((params['scale'] > 0).all()))

    def test_vib_kl_is_scalar_and_nonnegative(self):
        out = self.model(
            query=['c1', 'c2', 'c3'] + self.model.embedding_query_names,
            input=self.x,
        )
        kl = self.model.vib_kl(out)
        self.assertEqual(kl.shape, ())
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_vib_kl_requires_embedding_query(self):
        """
        The embeddings' parameters are only in the output if they were queried.
        """
        out = self.model(query=['c1', 'c2', 'c3'], input=self.x)
        with pytest.raises(ValueError):
            self.model.vib_kl(out)

    def test_concept_uncertainty_shape(self):
        out = self.model(
            query=self.model.embedding_query_names,
            input=self.x,
        )
        unc = self.model.concept_uncertainty(out)
        self.assertEqual(unc.shape, (8, 3))
        self.assertTrue(bool((unc > 0).all()))


class TestProbCBMInference(unittest.TestCase):
    """
    Deterministic (mean) versus ancestral (sampled) inference.
    """

    def test_deterministic_is_repeatable_sampling_is_not(self):
        model = _model()
        x = torch.randn(8, 6)
        model.eval()
        # Deterministic (mean) propagation is identical across calls
        a = model(query=['task'], input=x).params['task']['logits']
        b = model(query=['task'], input=x).params['task']['logits']
        self.assertTrue(torch.allclose(a, b))
        # While ancestral sampling draws the embeddings, so it varies
        model.setup_inference(inference=AncestralSamplingInference)
        c = model(query=['task'], input=x).params['task']['logits']
        d = model(query=['task'], input=x).params['task']['logits']
        self.assertFalse(torch.allclose(c, d))
        model.setup_inference(inference=DeterministicInference)

    def test_intervention_runs(self):
        """
        Interventions clamp the concept *embeddings* to their anchors, which is
        what :meth:`ProbCBM.anchor_embeddings` builds.
        """
        model = _model()
        x = torch.randn(8, 6)
        labels = torch.randint(0, 2, (8, 3)).float()
        evidence = {'input': x, **model.anchor_embeddings(labels)}
        out = model(query=['task'], evidence=evidence)
        self.assertEqual(out.params['task']['logits'].shape, (8, 1))

    def test_intervention_on_concepts_runs_when_interpolating(self):
        """
        With ``use_anchor_interpolation`` the concepts are the task's parents,
        so they can be clamped directly.
        """
        model = _model(use_anchor_interpolation=True)
        x = torch.randn(8, 6)
        evidence = {'input': x}
        for name in ['c1', 'c2', 'c3']:
            evidence[name] = torch.randint(0, 2, (8, 1)).float()
        out = model(query=['task'], evidence=evidence)
        self.assertEqual(out.params['task']['logits'].shape, (8, 1))


class TestProbCBMPaperEquations(unittest.TestCase):
    """
    Checks that the forward pass is a literal transcription of the equations in
    Kim et al. (2023). Each test recomputes an output from the model's own raw
    parameters using the paper's equation written out by hand, then asserts that
    the model agrees with it.
    """

    CONCEPTS = ['c1', 'c2', 'c3']
    EMBEDDING_SIZE = 8

    def setUp(self):
        torch.manual_seed(0)
        self.model = _model()
        self.model.eval()
        self.x = torch.randn(6, 6)
        with torch.no_grad():
            self.out = self.model(
                query=(
                    self.CONCEPTS +
                    ['task'] +
                    self.model.embedding_query_names
                ),
                input=self.x,
            )
        # Normalised anchors of shape (n_concepts, 2, embedding_size), where
        # index 0 is the negative anchor z^- and index 1 the positive one z^+.
        self.anchors = F.normalize(
            self.model.concept_predictors[0].anchors.anchors,
            p=2,
            dim=-1,
        )
        self.task_head = \
            self.model.pgm.factors['tasks'].parametrization['logits']

    def _embedding(self, quantity):
        """
        The queried embeddings' ``loc``/``scale``, reshaped per concept into
        ``(batch, n_concepts, embedding_size)``.
        """
        return torch.cat(
            [
                self.out.params[name][quantity].reshape(
                    6, -1, self.EMBEDDING_SIZE,
                )
                for name in self.model.embedding_query_names
            ],
            dim=1,
        )

    def _class_logits(self, concept_embeddings):
        """
        Eqs. 4-5 written by hand: project the concept embeddings into the
        class-embedding space, then take the distances to the class anchors.
        """
        h = self.model.class_projection(concept_embeddings.flatten(1))
        anchors = self.task_head.anchors.anchors[0]  # (2, D_y), binary task
        distance = torch.sqrt(
            (h.unsqueeze(1) - anchors.unsqueeze(0)).pow(2).mean(-1) + 1e-10
        )
        # For a binary task this is the single logit of the two-anchor softmax
        return self.task_head.anchors.scale * (distance[:, 0] - distance[:, 1])

    def test_embedding_means_lie_on_the_unit_sphere(self):
        norms = self._embedding('loc').norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        )

    def test_concept_logit_is_the_anchor_distance_difference(self):
        """
        Eq. 3: ``logit = a * (||z - z^-||_2 - ||z - z^+||_2)``.
        """
        z = self._embedding('loc')
        distance = torch.sqrt(
            (z.unsqueeze(2) - self.anchors.unsqueeze(0)).pow(2).sum(-1) + 1e-6
        )
        expected = self.model.concept_predictors[0].anchors.scale * (
            distance[..., 0] - distance[..., 1]
        )
        self.assertTrue(torch.allclose(
            self.out.logits[self.CONCEPTS], expected, atol=1e-5,
        ))

    def test_task_logit_is_the_class_anchor_distance(self):
        """
        Eqs. 4-5, evaluated on the predicted concept embeddings themselves.
        """
        expected = self._class_logits(self._embedding('loc'))
        self.assertTrue(torch.allclose(
            self.out.logits[['task']].squeeze(-1), expected, atol=1e-5,
        ))

    def test_interpolating_head_reads_the_concept_activations(self):
        """
        The opt-in variant swaps the predicted embedding for the interpolation
        ``v_i * z_i^+ + (1 - v_i) * z_i^-`` of the concept activations.
        """
        torch.manual_seed(0)
        model = _model(use_anchor_interpolation=True)
        model.eval()
        with torch.no_grad():
            out = model(query=self.CONCEPTS + ['task'], input=self.x)
            activations = torch.sigmoid(out.logits[self.CONCEPTS])
            head = model.pgm.factors['tasks'].parametrization['logits']
            h = model.class_projection(
                model.concept_predictors[0].anchors.interpolate(activations).flatten(1)
            )
            anchors = head.anchors.anchors[0]
            distance = torch.sqrt(
                (h.unsqueeze(1) - anchors.unsqueeze(0)).pow(2).mean(-1) + 1e-10
            )
            expected = head.anchors.scale * (distance[:, 0] - distance[:, 1])
        self.assertTrue(torch.allclose(
            out.logits[['task']].squeeze(-1), expected, atol=1e-5,
        ))

    def test_intervention_substitutes_the_ground_truth_anchor(self):
        """
        An intervention replaces the predicted embedding ``z_c`` by ``z_c^+``
        or ``z_c^-`` according to the label, which is the paper's semantics and
        what :meth:`ProbCBM.anchor_embeddings` produces.
        """
        labels = (torch.rand(6, len(self.CONCEPTS)) < 0.5).float()
        with torch.no_grad():
            out = self.model(
                query=['task'],
                evidence={
                    'input': self.x,
                    **self.model.anchor_embeddings(labels),
                },
            )
        # A value of 1 selects the positive anchor and 0 the negative one
        gt_embeddings = self.model.concept_predictors[0].anchors.interpolate(labels)
        self.assertTrue(torch.allclose(
            out.params['task']['logits'].squeeze(-1),
            self._class_logits(gt_embeddings),
            atol=1e-5,
        ))

    def test_partial_intervention_keeps_the_predicted_embeddings(self):
        """
        Concepts left out of an intervention keep the embedding the model
        predicted for them.
        """
        labels = (torch.rand(6, 1) < 0.5).float()
        with torch.no_grad():
            embeddings = self.model.anchor_embeddings(
                {'c1': labels},
                out=self.out,
            )['concepts__emb'].reshape(6, 3, self.EMBEDDING_SIZE)
        predicted = self._embedding('loc')
        interpolated = self.model.concept_predictors[0].anchors.interpolate(
            torch.cat([labels, torch.zeros(6, 2)], dim=1)
        )
        # c1 became its ground-truth anchor...
        self.assertTrue(torch.allclose(
            embeddings[:, 0], interpolated[:, 0], atol=1e-6,
        ))
        # ...while c2 and c3 are untouched
        self.assertTrue(torch.allclose(
            embeddings[:, 1:], predicted[:, 1:], atol=1e-6,
        ))

    def test_vib_kl_matches_the_analytic_kl(self):
        """
        Eq. 6: ``KL(N(mu, diag(sigma)) || N(0, I))``, meaned over the batch and
        the concepts.
        """
        loc, scale = self._embedding('loc'), self._embedding('scale')
        expected = kl_divergence(
            Normal(loc, scale),
            Normal(torch.zeros_like(loc), torch.ones_like(scale)),
        ).sum(-1).mean()
        self.assertTrue(torch.allclose(
            self.model.vib_kl(self.out), expected, atol=1e-5,
        ))

    def test_uncertainty_is_the_geometric_mean_of_the_variances(self):
        """
        Sec. 4.4: the determinant of the diagonal covariance, per concept.
        """
        variances = self._embedding('scale').pow(2)
        expected = variances.log().mean(-1).exp()
        self.assertTrue(torch.allclose(
            self.model.concept_uncertainty(self.out), expected, atol=1e-5,
        ))


class TestProbCBMTraining(unittest.TestCase):
    """
    Gradient flow and parameter updates.
    """

    def test_parameters_update(self):
        """
        A few (SGD) optimizer steps actually move the model's parameters.
        """
        model = _model()
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
            loss = loss_fn(out.logits[query], target)
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()

        self.assertLess(loss.item(), first)
        changed = [
            n for n, p in model.named_parameters()
            if not torch.allclose(p.detach(), before[n])
        ]
        self.assertTrue(changed, "no parameters changed during training")

    def test_anchor_scale_receives_gradient(self):
        """
        The learnable distance scales are trained end-to-end.
        """
        model = _model()
        model.train()
        out = model(query=['c1', 'c2', 'c3', 'task'], input=torch.randn(8, 6))
        out.logits[['c1', 'c2', 'c3', 'task']].sum().backward()
        head = model.pgm.factors['tasks'].parametrization['logits']
        self.assertIsNotNone(model.concept_predictors[0].anchors.scale.grad)
        self.assertIsNotNone(head.anchors.scale.grad)


class TestProbCBMLightningRecipe(unittest.TestCase):
    """
    The training recipe the model carries when ``lightning=True``.

    Kim et al. train a ProbCBM in two stages, fitting the concept predictor
    first and the class predictor second with everything else frozen, hold the
    backbone frozen for a few warm-up epochs, replace predicted embeddings by
    ground-truth anchors while the class predictor trains, and give the parts
    learnt from scratch a larger learning rate than the pretrained backbone.
    These tests pin each of those down.
    """

    @staticmethod
    def _model(**kwargs):
        defaults = dict(
            input_size=6,
            annotations=_binary_ann(),
            task_names=['task'],
            embedding_size=4,
            class_embedding_size=8,
            backbone=MLP(input_size=6, hidden_size=8, n_layers=1),
            latent_size=8,
            concept_epochs=4,
            class_epochs=3,
            warm_epochs=2,
            lightning=True,
        )
        return ProbCBM(**{**defaults, **kwargs})

    def test_sequential_stage_schedule(self):
        model = self._model()
        self.assertEqual(model.total_epochs, 7)
        self.assertEqual(
            [model.training_stage(e) for e in range(model.total_epochs)],
            ['concept'] * 4 + ['class'] * 3,
        )

    def test_joint_mode_is_a_single_stage(self):
        model = self._model(train_class_mode='joint')
        self.assertEqual(model.total_epochs, 4)
        self.assertEqual(
            {model.training_stage(e) for e in range(model.total_epochs)},
            {'joint'},
        )

    def test_rejects_an_unknown_train_class_mode(self):
        with pytest.raises(ValueError):
            self._model(train_class_mode='alternating')

    def _trainable(self, model):
        """(backbone, class-predictor, everything else) trainable counts."""
        class_ids = {id(p) for p in model.class_parameters()}
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        counts = {'backbone': 0, 'class': 0, 'other': 0}
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in backbone_ids:
                counts['backbone'] += 1
            elif id(parameter) in class_ids:
                counts['class'] += 1
            else:
                counts['other'] += 1
        return counts

    def test_each_stage_freezes_the_other_side(self):
        """
        The concept stage leaves the class predictor alone and vice versa.
        """
        model = self._model()
        n_class = len(model.class_parameters())

        model.set_training_stage('concept', warm=False)
        counts = self._trainable(model)
        self.assertEqual(counts['class'], 0)
        self.assertGreater(counts['other'], 0)
        self.assertGreater(counts['backbone'], 0)

        model.set_training_stage('class', warm=False)
        counts = self._trainable(model)
        self.assertEqual(counts['class'], n_class)
        self.assertEqual(counts['other'], 0)
        self.assertEqual(counts['backbone'], 0)

    def test_joint_stage_trains_everything(self):
        model = self._model(train_class_mode='joint')
        model.set_training_stage('joint', warm=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_warmup_freezes_only_the_backbone(self):
        model = self._model()
        model.set_training_stage('concept', warm=True)
        self.assertEqual(self._trainable(model)['backbone'], 0)
        self.assertGreater(self._trainable(model)['other'], 0)

    def test_two_learning_rates_and_no_decay_on_the_anchors(self):
        model = self._model(lr=0.001, lr_ratio=10.0, weight_decay=0.1)
        groups = model.configure_optimizers()['optimizer'].param_groups

        backbone_ids = {id(p) for p in model.backbone.parameters()}
        for group in groups:
            is_backbone = id(group['params'][0]) in backbone_ids
            self.assertAlmostEqual(
                group['lr'], 0.001 if is_backbone else 0.01,
            )
        # The anchors and their scales must sit in a decay-free group
        decay_free = {
            id(p) for group in groups if group['weight_decay'] == 0.0
            for p in group['params']
        }
        for parameter in model._no_decay_parameters():
            self.assertIn(id(parameter), decay_free)

    def test_scheduler_spans_the_whole_run(self):
        model = self._model()
        scheduler = model.configure_optimizers()['lr_scheduler']
        self.assertEqual(scheduler.T_max, model.total_epochs)


class TestProbCBMLightningFit(unittest.TestCase):
    """
    A short end-to-end ``Trainer.fit``, which is what exercises the hooks the
    recipe hangs off (stage switching, warm-up, and the composed loss).
    """

    def test_fit_runs_both_stages_and_logs_each_loss_term(self):
        seed_everything(42)
        dataset = ToyDataset(dataset='xor', seed=42, n_gen=256)
        datamodule = ConceptDataModule(
            dataset=dataset,
            batch_size=128,
            val_size=0.2,
            test_size=0.2,
            seed=42,
        )
        datamodule.setup()

        n_features = dataset.input_data.shape[1]
        model = ProbCBM(
            input_size=n_features,
            annotations=dataset.annotations,
            task_names=['xor'],
            embedding_size=4,
            class_embedding_size=8,
            backbone=MLP(input_size=n_features, hidden_size=8, n_layers=1),
            latent_size=8,
            train_inference=AncestralSamplingInference,
            concept_epochs=2,
            class_epochs=1,
            warm_epochs=1,
            lightning=True,
        )
        trainer = Trainer(
            max_epochs=model.total_epochs,
            gradient_clip_val=2.0,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=datamodule)

        logged = {str(k) for k in trainer.logged_metrics}
        # Both terms are reported throughout, even while only one is optimised
        self.assertIn('train_concept_loss', logged)
        self.assertIn('train_class_loss', logged)
        self.assertIn('train_loss', logged)
        # The run ended in the class stage, so only the class side is trainable
        self.assertEqual(
            model.training_stage(model.total_epochs - 1), 'class',
        )
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        class_ids = {id(p) for p in model.class_parameters()}
        for name, parameter in model.named_parameters():
            if name in trainable:
                self.assertIn(id(parameter), class_ids, name)

    def test_p_replace_is_only_on_while_the_class_predictor_trains(self):
        """
        Forcing the embeddings would hand the concept loss its own answer, so
        it is confined to the stage that has no concept loss.
        """
        model = TestProbCBMLightningRecipe._model(
            train_inference=DeterministicInference,
            intervention_prob=0.5,
        )
        for epoch, expected in ((0, 0.0), (3, 0.0), (4, 0.5), (6, 0.5)):
            model.trainer = types.SimpleNamespace(current_epoch=epoch)
            model.on_train_epoch_start()
            self.assertEqual(model.train_inference.p_int, expected, f"@{epoch}")


if __name__ == '__main__':
    unittest.main()
