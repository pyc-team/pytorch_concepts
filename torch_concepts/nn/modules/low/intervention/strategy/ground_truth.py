import torch

from ...base.intervention import BaseConceptInterventionStrategy


class GroundTruthIntervention(BaseConceptInterventionStrategy):
    """
    Intervention that replaces predicted concepts with ground truth values.

    Implements do(C=c_true) operations by mixing predicted and ground truth
    concept values based on a binary mask.

    Args:
        ground_truth: Ground truth concept values of shape (batch_size, n_concepts).
    """

    def __init__(self, ground_truth: torch.Tensor):
        super().__init__()
        self.ground_truth = ground_truth

    def forward(self, *args, **kwargs):
        return self.ground_truth
