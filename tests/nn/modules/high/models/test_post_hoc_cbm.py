"""
Tests for the Post-hoc Concept Bottleneck Model (PCBM / PCBM-h).

A PostHocCBM (Yuksekgonul et al., ICLR 2023) freezes a pretrained backbone,
turns each concept into a concept-activation vector (CAV) whose score is the
normalised signed distance of the embedding to the concept hyperplane, and
trains only a sparse linear head to the tasks. The hybrid PCBM-h adds a linear
residual over the raw embedding, fitted sequentially. These tests cover:

- Construction, the binary-concept restriction, and the frozen backbone
- The PGM structure: deterministic (``Delta``) concept scores + task head
- The shared CAV bank (frozen buffer vs trainable parameter, pre-fitted vectors)
- The concept score equals the normalised signed distance to the CAV hyperplane
- The elastic-net regulariser
- The PCBM-h residual toggle and the sequential-fitting freeze
- Concept interventions and training
"""
import math
import pytest
import unittest

import torch
import torch.nn as nn

from torch_concepts.annotations import Annotations
from torch_concepts.distributions import Delta
from torch_concepts.nn import MLP, PostHocCBM
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.mid.variable import ConceptVariable


def _binary_ann(concepts=('c1', 'c2', 'c3'), task='task'):
    labels = list(concepts) + [task]
    return Annotations(
        labels=labels,
        cardinalities=[1] * len(labels),
        types=['binary'] * len(labels),
    )


def _model(ann=None, latent=16, **kwargs):
    ann = ann or _binary_ann()
    trunk = MLP(input_size=8, hidden_size=latent, n_layers=1)
    return PostHocCBM(
        input_size=8, annotations=ann, task_names=['task'],
        backbone=trunk, latent_size=latent, **kwargs,
    )


class TestPostHocCBMInit(unittest.TestCase):
    def test_basic_init(self):
        model = _model()
        self.assertFalse(isinstance(model, BaseLearner))
        self.assertEqual(model.cavs.n_concepts, 3)
        self.assertEqual(model.cavs.embedding_size, 16)

    def test_rejects_non_binary_concepts(self):
        ann = Annotations(
            labels=['c1', 'c2', 'task'],
            cardinalities=[3, 1, 1],
            types=['categorical', 'binary', 'binary'],
        )
        with pytest.raises(ValueError):
            PostHocCBM(input_size=8, annotations=ann, task_names=['task'])

    def test_backbone_frozen_by_default(self):
        model = _model()
        self.assertTrue(
            all(not p.requires_grad for p in model.backbone.parameters())
        )

    def test_backbone_trainable_when_requested(self):
        model = _model(freeze_backbone=False)
        self.assertTrue(
            any(p.requires_grad for p in model.backbone.parameters())
        )


class TestPostHocCBMCAVBank(unittest.TestCase):
    def test_frozen_cavs_are_buffers(self):
        model = _model(freeze_concept_vectors=True)
        self.assertNotIsInstance(model.cavs.vectors, nn.Parameter)
        self.assertNotIsInstance(model.cavs.intercepts, nn.Parameter)

    def test_trainable_cavs_are_parameters(self):
        model = _model(freeze_concept_vectors=False)
        self.assertIsInstance(model.cavs.vectors, nn.Parameter)
        self.assertIsInstance(model.cavs.intercepts, nn.Parameter)

    def test_prefitted_vectors_are_used(self):
        vectors = torch.randn(3, 16)
        intercepts = torch.randn(3)
        model = _model(concept_vectors=vectors, concept_intercepts=intercepts)
        self.assertTrue(torch.allclose(model.cavs.vectors, vectors))
        self.assertTrue(torch.allclose(model.cavs.intercepts, intercepts))

    def test_score_is_normalised_signed_distance(self):
        """The concept score equals ``(f(x).v + b) / ||v||``."""
        vectors = torch.randn(3, 16)
        intercepts = torch.randn(3)
        model = _model(concept_vectors=vectors, concept_intercepts=intercepts)
        model.eval()
        x = torch.randn(5, 8)
        out = model(query=['c1', 'c2', 'c3'], input=x)
        scores = out.value[['c1', 'c2', 'c3']]

        with torch.no_grad():
            emb = model.backbone(x)
            norm = vectors.norm(dim=-1)
            expected = emb @ (vectors / norm.unsqueeze(-1)).t() + intercepts / norm
        self.assertTrue(torch.allclose(scores, expected, atol=1e-5))


class TestPostHocCBMStructure(unittest.TestCase):
    def test_plate_layout_scores_are_delta(self):
        model = _model()
        variables = model.pgm.variables
        self.assertEqual(
            set(variables), {'input', 'latent', 'concepts', 'tasks'},
        )
        self.assertIsInstance(variables['concepts'], ConceptVariable)
        self.assertIs(variables['concepts'].distribution, Delta)

    def test_individual_layout(self):
        model = _model(plate=False)
        concept_vars = [
            v for v in model.pgm.variables.values()
            if v.variable_type == 'concept' and v.name != 'task'
        ]
        self.assertEqual({v.name for v in concept_vars}, {'c1', 'c2', 'c3'})
        self.assertTrue(all(v.distribution is Delta for v in concept_vars))

    def test_task_head_width_matches_all_concepts(self):
        model = _model()
        head = model.pgm.factors['tasks'].parametrization['logits']
        self.assertEqual(head.predictor.in_features, 3)


class TestPostHocCBMResidual(unittest.TestCase):
    def test_residual_changes_output(self):
        model = _model(residual=True)
        x = torch.randn(8, 8)
        model.set_residual_use(False)
        a = model(query=['task'], input=x).params['task']['logits']
        model.set_residual_use(True)
        b = model(query=['task'], input=x).params['task']['logits']
        self.assertFalse(torch.allclose(a, b))

    def test_freeze_non_residual_leaves_only_residual_trainable(self):
        model = _model(residual=True, freeze_concept_vectors=False)
        model.freeze_non_residual_components()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        # Every trainable tensor must belong to a residual head.
        self.assertTrue(trainable)
        self.assertTrue(all('residual' in n for n in trainable), trainable)

    def test_residual_false_toggle_is_noop(self):
        model = _model(residual=False)
        x = torch.randn(4, 8)
        model.set_residual_use(False)  # no residual heads -> nothing happens
        out = model(query=['task'], input=x)
        self.assertEqual(out.params['task']['logits'].shape, (4, 1))


class TestPostHocCBMRegularisation(unittest.TestCase):
    def test_elastic_net_is_scalar_nonnegative(self):
        model = _model()
        reg = model.elastic_net()
        self.assertEqual(reg.shape, ())
        self.assertGreaterEqual(reg.item(), 0.0)

    def test_elastic_net_zero_weights(self):
        model = _model()
        with torch.no_grad():
            for head in model._interpretable_heads:
                head.weight.zero_()
        self.assertEqual(model.elastic_net().item(), 0.0)


class TestPostHocCBMTraining(unittest.TestCase):
    def test_intervention_runs(self):
        model = _model()
        x = torch.randn(8, 8)
        evidence = {'input': x}
        for name in ['c1', 'c2', 'c3']:
            evidence[name] = (2.0 * torch.randint(0, 2, (8, 1)).float() - 1.0)
        out = model(query=['task'], evidence=evidence)
        self.assertEqual(out.params['task']['logits'].shape, (8, 1))

    def test_head_parameters_update(self):
        model = _model(freeze_concept_vectors=False)
        model.train()
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.5,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        x = torch.randn(16, 8)
        target = torch.randint(0, 2, (16, 1)).float()

        first = None
        for _ in range(30):
            opt.zero_grad()
            out = model(query=['task'], input=x)
            loss = loss_fn(out.params['task']['logits'], target) + model.elastic_net()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), first)


if __name__ == '__main__':
    unittest.main()
