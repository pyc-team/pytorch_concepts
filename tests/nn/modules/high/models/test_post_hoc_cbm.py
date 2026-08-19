"""
Set of tests for the Post-hoc Concept Bottleneck Model (PCBM / PCBM-h).

Everything shared with the plain CBM is already covered by ``test_cbm.py``, so
these tests cover only what is specific to reading concepts off a frozen
backbone through a bank of concept-activation vectors:
- Construction, the binary-concept restriction, and the frozen backbone
  (including the warning raised when we freeze someone else's backbone)
- The PGM structure: deterministic (``Delta``) concept scores + a task head
- The shared CAV bank (frozen buffer vs trainable parameter, pre-fitted
  vectors)
- That the concept score is the normalised signed distance to the CAV
  hyperplane
- The elastic-net regulariser (the paper's Eq. 1, checked numerically)
- The PCBM-h residual toggle and the sequential-fitting freeze
- Concept interventions: that clamping a score reaches the task head, and that
  intervening *improves task accuracy*
- Training, including the ``lightning=True`` recipe: the two sequential stages
  and what each freezes, and the penalty that rides along with each
"""
import pytest
import unittest
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.annotations import Annotations
from torch_concepts.data import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.distributions import Delta
from torch_concepts.nn import MLP, PostHocCBM
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.mid.variable import ConceptVariable

import intervention_benchmark as bench


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


def _model(ann=None, latent=16, **kwargs):
    """
    A small PostHocCBM whose "pretrained" backbone is an untrained MLP trunk.
    """
    ann = ann or _binary_ann()
    # Tests that care about how the backbone is treated pass their own.
    trunk = kwargs.pop(
        'backbone', MLP(input_size=8, hidden_size=latent, n_layers=1)
    )
    return PostHocCBM(
        input_size=8,
        annotations=ann,
        task_names=['task'],
        backbone=trunk,
        latent_size=latent,
        **kwargs,
    )


class TestPostHocCBMInit(unittest.TestCase):
    """
    Construction, concept-type validation, and the frozen backbone.
    """

    def test_basic_init(self):
        model = _model()
        self.assertFalse(isinstance(model, BaseLearner))
        self.assertEqual(model.concept_encoders[0].out_concepts_shape, 3)
        self.assertEqual(model.concept_encoders[0].in_embeddings_shape, 16)

    def test_rejects_non_binary_concepts(self):
        ann = Annotations(
            labels=['c1', 'c2', 'task'],
            cardinalities=[3, 1, 1],
            types=['categorical', 'binary', 'binary'],
        )
        with pytest.raises(ValueError):
            PostHocCBM(
                input_size=8,
                annotations=ann,
                task_names=['task'],
            )

    def test_backbone_frozen_by_default(self):
        """
        Not retraining the backbone is the whole point of the post-hoc setting.
        """
        model = _model()
        self.assertTrue(
            all(not p.requires_grad for p in model.backbone.parameters())
        )

    def test_backbone_trainable_when_requested(self):
        model = _model(freeze_backbone=False)
        self.assertTrue(
            any(p.requires_grad for p in model.backbone.parameters())
        )

    def test_warns_when_freezing_a_live_backbone(self):
        """
        The backbone is frozen in place, so a caller sharing it with another
        model needs to hear about it.
        """
        trunk = MLP(input_size=8, hidden_size=16, n_layers=1)
        with pytest.warns(UserWarning, match="freezing them in place"):
            _model(backbone=trunk)
        self.assertTrue(all(not p.requires_grad for p in trunk.parameters()))

    def test_silent_when_the_backbone_is_already_frozen(self):
        """
        Nothing is being changed, so there is nothing to warn about.
        """
        trunk = MLP(input_size=8, hidden_size=16, n_layers=1)
        for parameter in trunk.parameters():
            parameter.requires_grad_(False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _model(backbone=trunk)

    def test_silent_when_not_freezing_at_all(self):
        trunk = MLP(input_size=8, hidden_size=16, n_layers=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _model(backbone=trunk, freeze_backbone=False)
        self.assertTrue(all(p.requires_grad for p in trunk.parameters()))


class TestPostHocCBMCAVBank(unittest.TestCase):
    """
    The concept-activation vector bank and the scores it produces.
    """

    def test_frozen_cavs_are_buffers(self):
        model = _model(freeze_concept_vectors=True)
        self.assertNotIsInstance(model.concept_encoders[0].cavs, nn.Parameter)
        self.assertNotIsInstance(model.concept_encoders[0].bias, nn.Parameter)

    def test_trainable_cavs_are_parameters(self):
        model = _model(freeze_concept_vectors=False)
        self.assertIsInstance(model.concept_encoders[0].cavs, nn.Parameter)
        self.assertIsInstance(model.concept_encoders[0].bias, nn.Parameter)

    def test_prefitted_vectors_are_used(self):
        vectors = torch.randn(3, 16)
        intercepts = torch.randn(3)
        model = _model(
            concept_vectors=vectors,
            concept_intercepts=intercepts,
        )
        self.assertTrue(torch.allclose(model.concept_encoders[0].cavs, vectors))
        self.assertTrue(torch.allclose(model.concept_encoders[0].bias, intercepts))

    def test_score_is_normalised_signed_distance(self):
        """
        The concept score equals the geometric margin ``(f(x).v + b) / ||v||``.
        """
        vectors = torch.randn(3, 16)
        intercepts = torch.randn(3)
        model = _model(
            concept_vectors=vectors,
            concept_intercepts=intercepts,
        )
        model.eval()
        x = torch.randn(5, 8)
        out = model(query=['c1', 'c2', 'c3'], input=x)
        scores = out.value[['c1', 'c2', 'c3']]

        with torch.no_grad():
            emb = model.backbone(x)
            norm = vectors.norm(dim=-1)
            expected = (
                emb @ (vectors / norm.unsqueeze(-1)).t() + intercepts / norm
            )
        self.assertTrue(torch.allclose(scores, expected, atol=1e-5))


class TestPostHocCBMStructure(unittest.TestCase):
    """
    The assembled PGM: variable kinds, plate grouping, task head width.
    """

    def test_plate_layout_scores_are_delta(self):
        model = _model()
        variables = model.pgm.variables
        self.assertEqual(
            set(variables),
            {'input', 'latent', 'concepts', 'tasks'},
        )
        self.assertIsInstance(variables['concepts'], ConceptVariable)
        self.assertIs(variables['concepts'].distribution, Delta)

    def test_individual_layout(self):
        """
        ``plate=False`` builds one score variable per concept.
        """
        model = _model(plate=False)
        concept_vars = [
            v for v in model.pgm.variables.values()
            if (v.variable_type == 'concept') and (v.name != 'task')
        ]
        self.assertEqual({v.name for v in concept_vars}, {'c1', 'c2', 'c3'})
        self.assertTrue(all(v.distribution is Delta for v in concept_vars))

    def test_task_head_width_matches_all_concepts(self):
        model = _model()
        head = model.pgm.factors['tasks'].parametrization['logits']
        self.assertEqual(head.predictor.in_features, 3)


class TestPostHocCBMResidual(unittest.TestCase):
    """
    The PCBM-h residual: its toggle and the sequential-fitting freeze.
    """

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
        # Every trainable tensor must belong to a residual head
        self.assertTrue(trainable)
        self.assertTrue(all('residual' in n for n in trainable), trainable)

    def test_residual_false_toggle_is_noop(self):
        model = _model(residual=False)
        x = torch.randn(4, 8)
        model.set_residual_use(False)  # No residual heads -> nothing happens
        out = model(query=['task'], input=x)
        self.assertEqual(out.params['task']['logits'].shape, (4, 1))


class TestPostHocCBMRegularisation(unittest.TestCase):
    """
    The elastic-net penalty that keeps the interpretable head sparse.
    """

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

    def test_elastic_net_matches_the_paper_formula(self):
        """
        Eq. 1: ``lam / (Nc * K) * (alpha * ||W||_1 + (1 - alpha) * ||W||_2^2)``,
        where the L2 term is the *squared* norm and the whole penalty is
        normalised by the number of concepts times the number of task outputs.
        """
        model = _model()
        weight = model._interpretable_heads[0].weight
        expected = model.reg_strength * (
            model.l1_ratio * weight.abs().sum() +
            (1.0 - model.l1_ratio) * weight.pow(2).sum()
        ) / (3 * 1)  # 3 concepts, 1 binary task output
        self.assertTrue(
            torch.allclose(model.elastic_net(), expected, atol=1e-8)
        )


class TestPostHocCBMTraining(unittest.TestCase):
    """
    Interventions on the concept scores, and parameter updates.
    """

    def test_intervention_runs(self):
        model = _model()
        x = torch.randn(8, 8)
        evidence = {'input': x}
        for name in ['c1', 'c2', 'c3']:
            evidence[name] = (2.0 * torch.randint(0, 2, (8, 1)).float() - 1.0)
        out = model(query=['task'], evidence=evidence)
        self.assertEqual(out.params['task']['logits'].shape, (8, 1))

    def test_intervention_clamps_the_score_reaching_the_task_head(self):
        """
        The task logits are exactly the head applied to the clamped scores.
        """
        model = _model()
        model.eval()
        x = torch.randn(8, 8)
        clamped = 2.0 * (torch.rand(8, 3) < 0.5).float() - 1.0
        evidence = {'input': x}
        for i, name in enumerate(['c1', 'c2', 'c3']):
            evidence[name] = clamped[:, i:i + 1]
        with torch.no_grad():
            out = model(query=['task'], evidence=evidence)
            expected = model._task_heads[0].predictor(clamped)
        self.assertTrue(torch.allclose(
            out.params['task']['logits'], expected, atol=1e-6,
        ))

    def test_head_parameters_update(self):
        """
        A few (SGD) optimizer steps actually move the trainable parameters.
        """
        model = _model(freeze_concept_vectors=False)
        model.train()
        # SGD (not Adam) keeps this runnable on CPU-only boxes whose torch build
        # trips an accelerator health-check inside the fused Adam step.
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.5,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        x = torch.randn(16, 8)
        target = torch.randint(0, 2, (16, 1)).float()

        first = None
        for _ in range(30):
            opt.zero_grad()
            out = model(query=['task'], input=x)
            loss = (
                loss_fn(out.params['task']['logits'], target) +
                model.elastic_net()
            )
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()

        self.assertLess(loss.item(), first)


class TestPostHocCBMInterventionsImprovePerformance(unittest.TestCase):
    """
    Intervening on the concept scores must make the task prediction better.

    This runs the whole post-hoc recipe on the shared benchmark (see
    ``intervention_benchmark``): pretrain a black box on the task alone, freeze
    its trunk, fit one CAV per concept on the frozen embeddings, and then train
    only the sparse interpretable head. Because the concepts here are signed
    margins rather than probabilities, interventions clamp them to ``+/-1``,
    i.e. a unit margin on the correct side of the concept hyperplane.
    """

    @staticmethod
    def _pretrain_blackbox(x, y, epochs=400):
        """
        A trunk trained end-to-end on the task alone, with no concepts, which is
        then frozen and handed to the PostHocCBM as its pretrained backbone.
        """
        trunk = MLP(
            input_size=bench.N_CONCEPTS,
            hidden_size=bench.HIDDEN_SIZE,
            n_layers=2,
        )
        head = nn.Linear(bench.HIDDEN_SIZE, 1)
        opt = torch.optim.SGD(
            list(trunk.parameters()) + list(head.parameters()),
            lr=0.1,
            momentum=0.9,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(head(trunk(x)), y).backward()
            opt.step()
        return trunk

    @staticmethod
    def _fit_cavs(embeddings, concepts):
        """
        One logistic-regression probe per concept on the frozen embeddings, as
        in the original PCBM pipeline.
        """
        vectors, intercepts = [], []
        for i in range(concepts.shape[1]):
            probe = LogisticRegression(max_iter=1000).fit(
                embeddings,
                concepts[:, i].numpy(),
            )
            vectors.append(probe.coef_[0])
            intercepts.append(probe.intercept_[0])
        return (
            torch.tensor(np.stack(vectors)).float(),
            torch.tensor(np.stack(intercepts)).float(),
        )

    def test_task_accuracy_rises_with_the_number_of_intervened_concepts(self):
        torch.manual_seed(0)
        x_train, c_train, y_train = bench.make_data(1000, seed=0)
        x_test, c_test, y_test = bench.make_data(800, seed=1)

        trunk = self._pretrain_blackbox(x_train, y_train)
        with torch.no_grad():
            embeddings = trunk(x_train).numpy()
        vectors, intercepts = self._fit_cavs(embeddings, c_train)

        model = PostHocCBM(
            input_size=bench.N_CONCEPTS,
            annotations=bench.annotations(),
            task_names=[bench.TASK_NAME],
            concept_vectors=vectors,
            concept_intercepts=intercepts,
            backbone=trunk,
            latent_size=bench.HIDDEN_SIZE,
        )

        # In the post-hoc setting the concept side is fixed, so we only fit the
        # task head, under the elastic-net penalty that keeps it sparse.
        bench.train(
            model,
            x_train,
            c_train,
            y_train,
            task_only=True,
            extra_loss=model.elastic_net,
        )

        accuracies = bench.intervention_curve(
            model,
            x_test,
            c_test,
            y_test,
            value_fn=lambda i, c: 2.0 * c[:, i:i + 1] - 1.0,
        )
        bench.assert_interventions_help(self, accuracies)


class TestPostHocCBMLightningRecipe(unittest.TestCase):
    """
    The training recipe the model carries when ``lightning=True``.

    Yuksekgonul et al. fit a PCBM-h in two passes: the sparse interpretable
    head first, under the elastic net, and only then the residual, with
    everything else frozen and its own L2 penalty. These tests pin that down.
    """

    @staticmethod
    def _model(**kwargs):
        defaults = dict(
            residual=True,
            interpretable_epochs=3,
            residual_epochs=2,
            lightning=True,
        )
        return _model(**{**defaults, **kwargs})

    def test_stage_schedule(self):
        model = self._model()
        self.assertEqual(model.total_epochs, 5)
        self.assertEqual(
            [model.training_stage(e) for e in range(model.total_epochs)],
            ['interpretable'] * 3 + ['residual'] * 2,
        )

    def test_without_a_residual_there_is_only_one_stage(self):
        model = self._model(residual=False)
        self.assertEqual(model.total_epochs, model.interpretable_epochs)
        self.assertEqual(
            {model.training_stage(e) for e in range(model.total_epochs)},
            {'interpretable'},
        )

    def _trainable(self, model, parameters):
        ids = {id(p) for p in parameters}
        return sum(
            1 for p in model.parameters()
            if id(p) in ids and p.requires_grad
        )

    def test_interpretable_stage_holds_the_residual_off_and_frozen(self):
        model = self._model()
        head = model._task_heads[0]
        model.set_training_stage('interpretable')

        self.assertFalse(head.residual_use)
        self.assertEqual(self._trainable(model, head.residual.parameters()), 0)
        self.assertGreater(self._trainable(model, head.c2y.parameters()), 0)
        # The post-hoc setting keeps the backbone fixed throughout
        self.assertEqual(
            self._trainable(model, model.backbone.parameters()), 0,
        )

    def test_residual_stage_trains_only_the_residual(self):
        model = self._model()
        head = model._task_heads[0]
        model.set_training_stage('residual')

        self.assertTrue(head.residual_use)
        self.assertEqual(
            self._trainable(model, head.residual.parameters()),
            len(list(head.residual.parameters())),
        )
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all('residual' in n for n in trainable), trainable)

    def test_residual_penalty_is_the_mean_square_of_its_weights(self):
        model = self._model()
        expected = model._task_heads[0].residual.weight.pow(2).mean()
        self.assertTrue(
            torch.allclose(model._residual_penalty(), expected, atol=1e-8)
        )

    def test_residual_penalty_is_zero_without_a_residual(self):
        model = self._model(residual=False)
        self.assertEqual(model._residual_penalty().item(), 0.0)

    def test_optimizer_uses_the_configured_learning_rate(self):
        model = self._model(lr=0.005, weight_decay=0.1)
        group = model.configure_optimizers()['optimizer'].param_groups[0]
        self.assertAlmostEqual(group['lr'], 0.005)
        self.assertAlmostEqual(group['weight_decay'], 0.1)


class TestPostHocCBMLightningFit(unittest.TestCase):
    """
    A short end-to-end ``Trainer.fit``, exercising the stage hooks.
    """

    def test_fit_runs_both_stages_and_leaves_the_residual_on(self):
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
        model = PostHocCBM(
            input_size=n_features,
            annotations=dataset.annotations,
            task_names=['xor'],
            residual=True,
            backbone=MLP(input_size=n_features, hidden_size=8, n_layers=1),
            latent_size=8,
            interpretable_epochs=2,
            residual_epochs=1,
            lightning=True,
        )
        trainer = Trainer(
            max_epochs=model.total_epochs,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=datamodule)

        logged = {str(k) for k in trainer.logged_metrics}
        self.assertIn('train_task_loss', logged)
        self.assertIn('train_loss', logged)
        # The run ends in the residual stage, so the residual is on and is the
        # only thing still trainable.
        self.assertTrue(model._task_heads[0].residual_use)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all('residual' in n for n in trainable), trainable)


if __name__ == '__main__':
    unittest.main()
