"""Tests for loss composition and the probability-space concept term.

An objective is not always a single concept term, so it is built from
independent pieces: ``CompositeLoss`` sums them, each term reading only the
``ModelOutput`` it is handed, so no term is tied to a particular model.
``NLLProbLoss`` is the concept term for a head that reports ``probs`` rather
than logits, as every PGM forward pass does.
"""
import pytest
import torch
import torch.nn.functional as F

from torch_concepts.annotations import Annotations
from torch_concepts.nn import CompositeLoss, ConceptLoss, NLLProbLoss
from torch_concepts.nn.modules.loss import PyCLoss
from torch_concepts.nn.modules.outputs import ModelOutput
from torch_concepts.tensor import AnnotatedTensor

B = 8


def annotated(tensor, labels, cardinalities, types):
    return AnnotatedTensor(
        tensor, Annotations(labels=labels, cardinalities=cardinalities, types=types), 1
    )


class OutputOnlyTerm(PyCLoss):
    """A term whose ``forward`` takes the output alone, like
    ``WeightedConceptLoss`` — the other half of ``CompositeLoss``'s signature
    dispatch."""

    def forward(self, output: ModelOutput) -> torch.Tensor:
        return output.probs.tensor.abs().mean()


@pytest.fixture
def output():
    """A ModelOutput reporting ``probs`` for a supervised concept and for an
    unsupervised variable the target does not cover."""
    torch.manual_seed(0)
    n_extra = 12
    out = ModelOutput()
    out.probs = annotated(
        torch.cat([torch.rand(B, 2), torch.rand(B, n_extra)], dim=-1),
        ["c", "u"], [2, n_extra], ["categorical", "categorical"],
    )
    # The target is concept-space: one integer-coded column per concept.
    out.target = AnnotatedTensor(
        torch.randint(0, 2, (B, 1)),
        Annotations(
            labels=["c"], cardinalities=[2], types=["categorical"]
        ).to_concept_space(),
        1,
    )
    return out


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
        first, second = OutputOnlyTerm(), OutputOnlyTerm()
        total = CompositeLoss(terms=[first, second], weights=[2.0, 3.0])(output)
        assert torch.allclose(
            total, 2.0 * first(output) + 3.0 * second(output), atol=1e-5
        )

    def test_weights_default_to_one(self, output):
        term = OutputOnlyTerm()
        assert torch.allclose(
            CompositeLoss(terms=[term, term])(output), 2.0 * term(output), atol=1e-5
        )

    def test_terms_with_and_without_a_target_compose(self, output):
        # ConceptLoss.forward takes (output, target) but OutputOnlyTerm takes
        # only (output) — dispatch is by signature.
        concept = ConceptLoss(categorical=NLLProbLoss(), categorical_param="probs")
        extra = OutputOnlyTerm()
        loss = CompositeLoss(terms=[concept, extra], weights=[5.0, 0.5])
        assert torch.allclose(
            loss(output, output.target),
            5.0 * concept(output, output.target) + 0.5 * extra(output),
            atol=1e-5,
        )

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError, match="Number of weights"):
            CompositeLoss(terms=[OutputOnlyTerm()], weights=[1.0, 2.0])

    def test_an_empty_term_list_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            CompositeLoss(terms=[])

    def test_it_is_a_type_aware_loss_so_the_learner_accepts_it(self):
        assert isinstance(CompositeLoss(terms=[OutputOnlyTerm()]), PyCLoss)


class TestUnsupervisedVariablesAreSkipped:
    def test_concept_loss_ignores_variables_with_no_ground_truth(self, output):
        # `probs` spans both `c` and the unsupervised `u`; only `c` has a target,
        # so scoring must not go looking for `u` in it.
        loss = ConceptLoss(categorical=NLLProbLoss(), categorical_param="probs")
        assert loss(output, output.target).ndim == 0
