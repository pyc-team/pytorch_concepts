"""
Tests for the Probabilistic Concept Bottleneck Model (ProbCBM).

A ProbCBM (Kim et al., ICML 2023) represents every concept by a Gaussian
*embedding*, decodes concepts from distances to learnable positive/negative
anchors, and predicts tasks from distances to learnable class anchors. These
tests cover the behaviour that is specific to that construction (everything
shared with the plain CBM is covered by ``test_cbm.py``):

- Construction and the binary-concept restriction
- The PGM structure: one ``Normal`` embedding variable + one anchor-decoded
  concept variable per plate group, and the shared anchor / projection tables
- Both building layouts (a single plate vs one variable per concept)
- Forward pass / output shapes, categorical tasks
- The VIB regulariser and per-concept uncertainty helpers
- Deterministic (mean) vs ancestral (sampled) inference
- Concept interventions (clamping concepts substitutes their anchor embeddings)
- Gradient flow / parameter updates
"""
import pytest
import unittest

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    AncestralSamplingInference,
    DeterministicInference,
    ProbCBM,
)
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.mid.variable import (
    ConceptVariable,
    EmbeddingVariable,
)


def _binary_ann(concepts=('c1', 'c2', 'c3'), task='task'):
    labels = list(concepts) + [task]
    return Annotations(
        labels=labels,
        cardinalities=[1] * len(labels),
        types=['binary'] * len(labels),
    )


def _model(ann=None, **kwargs):
    ann = ann or _binary_ann()
    return ProbCBM(
        input_size=6, annotations=ann, task_names=['task'],
        embedding_size=8, class_embedding_size=16, **kwargs,
    )


class TestProbCBMInit(unittest.TestCase):
    def test_basic_init(self):
        model = _model()
        self.assertIsInstance(model.pgm, nn.Module)
        self.assertFalse(isinstance(model, BaseLearner))
        # The concept anchors and class projection share the same anchor table.
        self.assertIs(model.class_projection.anchors, model.concept_anchors)
        self.assertEqual(model.concept_anchors.n_concepts, 3)

    def test_lightning_mode(self):
        self.assertIsInstance(_model(lightning=True), BaseLearner)

    def test_rejects_non_binary_concepts(self):
        ann = Annotations(
            labels=['c1', 'c2', 'task'],
            cardinalities=[3, 1, 1],
            types=['categorical', 'binary', 'binary'],
        )
        with pytest.raises(ValueError):
            ProbCBM(input_size=6, annotations=ann, task_names=['task'])

    def test_categorical_task_is_allowed(self):
        ann = Annotations(
            labels=['c1', 'c2', 't'],
            cardinalities=[1, 1, 3],
            types=['binary', 'binary', 'categorical'],
        )
        model = ProbCBM(
            input_size=6, annotations=ann, task_names=['t'],
            embedding_size=8, class_embedding_size=16,
        )
        out = model(query=['t'], input=torch.randn(4, 6))
        self.assertEqual(out.params['t']['logits'].shape, (4, 3))


class TestProbCBMStructure(unittest.TestCase):
    def test_plate_layout_variables(self):
        model = _model()
        variables = model.pgm.variables
        # One batched Normal embedding + one concept plate + one task plate.
        self.assertEqual(
            set(variables), {'input', 'latent', 'concepts__emb', 'concepts', 'tasks'},
        )
        self.assertIsInstance(variables['concepts__emb'], EmbeddingVariable)
        self.assertIs(variables['concepts__emb'].distribution, Normal)
        self.assertIsInstance(variables['concepts'], ConceptVariable)
        self.assertIs(variables['concepts'].distribution, Bernoulli)
        # The embedding event stacks (n_concepts, embedding_size).
        self.assertEqual(tuple(variables['concepts__emb'].shape), (3, 8))

    def test_individual_layout_variables(self):
        model = _model(plate=False)
        emb = [v for v in model.pgm.variables.values()
               if v.variable_type == 'embedding' and v.name not in ('input', 'latent')]
        self.assertEqual({v.name for v in emb}, {'c1__emb', 'c2__emb', 'c3__emb'})
        for v in emb:
            self.assertEqual(tuple(v.shape), (8,))

    def test_embedding_query_names(self):
        self.assertEqual(_model().embedding_query_names, ['concepts__emb'])
        self.assertEqual(
            set(_model(plate=False).embedding_query_names),
            {'c1__emb', 'c2__emb', 'c3__emb'},
        )


class TestProbCBMForward(unittest.TestCase):
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
        # The location head normalises each concept embedding onto the sphere.
        loc = params['loc'].reshape(8, 3, 8)
        norms = loc.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        # Scale is strictly positive (Softplus).
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
        out = self.model(query=['c1', 'c2', 'c3'], input=self.x)
        with pytest.raises(ValueError):
            self.model.vib_kl(out)

    def test_concept_uncertainty_shape(self):
        out = self.model(query=self.model.embedding_query_names, input=self.x)
        unc = self.model.concept_uncertainty(out)
        self.assertEqual(unc.shape, (8, 3))
        self.assertTrue(bool((unc > 0).all()))


class TestProbCBMInference(unittest.TestCase):
    def test_deterministic_is_repeatable_sampling_is_not(self):
        model = _model()
        x = torch.randn(8, 6)
        model.eval()
        # Deterministic (mean) propagation: identical across calls.
        a = model(query=['task'], input=x).params['task']['logits']
        b = model(query=['task'], input=x).params['task']['logits']
        self.assertTrue(torch.allclose(a, b))
        # Ancestral sampling: embeddings/concepts are sampled -> varies.
        model.setup_inference(inference=AncestralSamplingInference)
        c = model(query=['task'], input=x).params['task']['logits']
        d = model(query=['task'], input=x).params['task']['logits']
        self.assertFalse(torch.allclose(c, d))
        model.setup_inference(inference=DeterministicInference)

    def test_intervention_runs(self):
        """Clamping concepts to evidence substitutes their anchor embeddings."""
        model = _model()
        x = torch.randn(8, 6)
        evidence = {'input': x}
        for name in ['c1', 'c2', 'c3']:
            evidence[name] = torch.randint(0, 2, (8, 1)).float()
        out = model(query=['task'], evidence=evidence)
        self.assertEqual(out.params['task']['logits'].shape, (8, 1))


class TestProbCBMTraining(unittest.TestCase):
    def test_parameters_update(self):
        model = _model()
        model.train()
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
        self.assertTrue(changed)

    def test_anchor_scale_receives_gradient(self):
        """The learnable distance scales are trained end-to-end."""
        model = _model()
        model.train()
        out = model(query=['c1', 'c2', 'c3', 'task'], input=torch.randn(8, 6))
        out.logits[['c1', 'c2', 'c3', 'task']].sum().backward()
        self.assertIsNotNone(model.concept_anchors.negative_scale.grad)
        self.assertIsNotNone(model.class_projection.scale.grad)


if __name__ == '__main__':
    unittest.main()
