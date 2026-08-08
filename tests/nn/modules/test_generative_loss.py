"""Tests for the composable generative loss terms.

An ELBO is not a single concept term, so it is built from independent pieces:
``CompositeLoss`` sums them, and each of ``ReconstructionLoss``,
``KLDivergenceLoss``, ``OrthogonalityLoss`` reads only a ``ModelOutput`` — so
none of them is tied to a particular model.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    CompositeLoss,
    ConceptLoss,
    KLDivergenceLoss,
    NLLProbLoss,
    OrthogonalityLoss,
    ReconstructionLoss,
)
from torch_concepts.nn.functional import concept_orthogonality
from torch_concepts.nn.modules.outputs import ModelOutput
from torch_concepts.tensor import AnnotatedTensor

B = 8


def annotated(tensor, labels, cardinalities, types):
    return AnnotatedTensor(
        tensor, Annotations(labels=labels, cardinalities=cardinalities, types=types), 1
    )


@pytest.fixture
def output():
    """A ModelOutput shaped like a generative model's: an observation plus
    concepts under ``probs``, a latent under ``loc``/``scale`` in both the model
    and the guide, and a Delta bottleneck under ``value``."""
    torch.manual_seed(0)
    n_pixels = 12
    out = ModelOutput()
    out.probs = annotated(
        torch.cat([torch.rand(B, 2), torch.rand(B, n_pixels)], dim=-1),
        ["c", "input"], [2, n_pixels], ["categorical", "categorical"],
    )
    out.loc = annotated(torch.randn(B, 4), ["z"], [4], ["continuous"])
    out.scale = annotated(torch.rand(B, 4) + 0.5, ["z"], [4], ["continuous"])
    out.guide_params["loc"] = annotated(
        torch.randn(B, 4), ["z"], [4], ["continuous"]
    )
    out.guide_params["scale"] = annotated(
        torch.rand(B, 4) + 0.5, ["z"], [4], ["continuous"]
    )
    # The bottleneck: one mixed context per concept (here 1, of width 3) followed
    # by the unsupervised one, so the width is (n_concepts + 1) * embedding_size.
    out.value = annotated(
        torch.randn(B, 6), ["mixing", "unknown"], [3, 3],
        ["continuous", "continuous"],
    )
    # The target is concept-space: one integer-coded column per concept.
    out.target = AnnotatedTensor(
        torch.randint(0, 2, (B, 1)),
        Annotations(
            labels=["c"], cardinalities=[2], types=["categorical"]
        ).to_concept_space(),
        1,
    )
    out.extra = {"evidence": {"input": torch.rand(B, n_pixels)}}
    return out


class TestReconstructionLoss:
    def test_it_equals_the_binary_cross_entropy_of_the_observation(self, output):
        mine = ReconstructionLoss(variable="input")(output)
        reference = F.binary_cross_entropy(
            output.probs["input"].tensor,
            output.extra["evidence"]["input"],
            reduction="none",
        ).sum(-1).mean()
        assert torch.allclose(mine, reference, atol=1e-5)

    def test_it_accepts_an_observation_that_kept_its_event_shape(self, output):
        # An image arrives as (B, C, H, W) but the parameters are flat.
        output.extra["evidence"]["input"] = torch.rand(B, 3, 2, 2)
        assert ReconstructionLoss(variable="input")(output).ndim == 0

    def test_a_gaussian_observation_uses_a_gaussian_likelihood(self):
        out = ModelOutput()
        out.loc = annotated(torch.zeros(B, 3), ["y"], [3], ["continuous"])
        out.scale = annotated(torch.ones(B, 3), ["y"], [3], ["continuous"])
        observed = torch.randn(B, 3)
        out.extra = {"evidence": {"y": observed}}
        expected = -torch.distributions.Independent(
            Normal(torch.zeros(B, 3), torch.ones(B, 3)), 1
        ).log_prob(observed).mean()
        assert torch.allclose(ReconstructionLoss(variable="y")(out), expected, atol=1e-5)

    def test_a_missing_observation_is_a_clear_error(self, output):
        output.extra = {}
        with pytest.raises(ValueError, match="no observed value"):
            ReconstructionLoss(variable="input")(output)


class TestKLDivergenceLoss:
    def test_it_matches_the_closed_form_kl(self, output):
        mine = KLDivergenceLoss(latents=["z"])(output)
        q = Normal(output.guide_params["loc"]["z"].tensor,
                   output.guide_params["scale"]["z"].tensor)
        p = Normal(output.loc["z"].tensor, output.scale["z"].tensor)
        assert torch.allclose(mine, kl_divergence(q, p).sum(-1).mean(), atol=1e-5)

    def test_an_identical_guide_and_prior_give_zero(self, output):
        output.guide_params["loc"] = output.params["loc"]
        output.guide_params["scale"] = output.params["scale"]
        assert torch.allclose(
            KLDivergenceLoss(latents=["z"])(output), torch.zeros(()), atol=1e-6
        )


class TestOrthogonalityLoss:
    def test_it_wraps_concept_orthogonality(self, output):
        mine = OrthogonalityLoss(variables=["mixing", "unknown"])(output)
        context = torch.cat(
            [output.value["mixing"], output.value["unknown"]], dim=-1
        )
        assert torch.allclose(mine, concept_orthogonality(context, 1), atol=1e-5)

    def test_n_concepts_can_be_given_explicitly(self, output):
        assert OrthogonalityLoss(
            variables=["mixing", "unknown"], n_concepts=2
        )(output).ndim == 0


class TestNLLProbLoss:
    def test_it_scores_probabilities_where_cross_entropy_would_need_logits(self):
        probs = torch.softmax(torch.randn(B, 5), -1)
        target = torch.randint(0, 5, (B,))
        assert torch.allclose(
            NLLProbLoss()(probs, target),
            F.nll_loss(probs.clamp_min(1e-8).log(), target),
        )

    def test_a_zero_probability_does_not_become_infinite(self):
        probs = torch.zeros(2, 3)
        probs[:, 0] = 1.0
        assert torch.isfinite(NLLProbLoss()(probs, torch.tensor([1, 2])))


class TestCompositeLoss:
    def test_it_is_the_weighted_sum_of_its_terms(self, output):
        recon, kl = ReconstructionLoss("input"), KLDivergenceLoss(["z"])
        total = CompositeLoss(terms=[recon, kl], weights=[2.0, 3.0])(output)
        assert torch.allclose(total, 2.0 * recon(output) + 3.0 * kl(output), atol=1e-5)

    def test_weights_default_to_one(self, output):
        recon, kl = ReconstructionLoss("input"), KLDivergenceLoss(["z"])
        assert torch.allclose(
            CompositeLoss(terms=[recon, kl])(output),
            recon(output) + kl(output), atol=1e-5,
        )

    def test_terms_with_and_without_a_target_compose(self, output):
        # ConceptLoss.forward takes (output, target); ReconstructionLoss does too,
        # but WeightedConceptLoss takes only (output) — dispatch is by signature.
        loss = CompositeLoss(
            terms=[
                ReconstructionLoss("input"),
                ConceptLoss(categorical=NLLProbLoss(), categorical_param="probs"),
            ],
            weights=[1.0, 5.0],
        )
        assert loss(output, output.target).ndim == 0

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError, match="Number of weights"):
            CompositeLoss(terms=[ReconstructionLoss()], weights=[1.0, 2.0])

    def test_it_is_a_type_aware_loss_so_the_learner_accepts_it(self, output):
        from torch_concepts.nn.modules.loss import TypeAwareLoss
        assert isinstance(CompositeLoss(terms=[ReconstructionLoss()]), TypeAwareLoss)


class TestUnsupervisedVariablesAreSkipped:
    def test_concept_loss_ignores_variables_with_no_ground_truth(self, output):
        # `probs` spans both `c` and the reconstructed `input`; only `c` has a
        # target, so scoring must not go looking for `input` in it.
        loss = ConceptLoss(categorical=NLLProbLoss(), categorical_param="probs")
        assert loss(output, output.target).ndim == 0
