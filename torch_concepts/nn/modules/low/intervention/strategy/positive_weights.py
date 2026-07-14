import torch

from ...base.intervention import BaseModuleInterventionStrategy


class PositiveWeightsIntervention(BaseModuleInterventionStrategy):
    """
    Intervention that replaces predicted concepts with ground truth values.

    Implements do(C=c_true) operations by mixing predicted and ground truth
    concept values based on a binary mask.

    Args:
        ground_truth: Ground truth concept values of shape (batch_size, n_concepts).
    """

    def __init__(self):
        super().__init__()

    def transform(self, module, *args, **kwargs):
        # find all parameters in the module and apply ReLU to them
        for name, param in module.named_parameters():
            with torch.no_grad():
                param.copy_(torch.relu(param))
        return module
