import torch

from ...base.intervention import BaseConceptInterventionStrategy


class DoIntervention(BaseConceptInterventionStrategy):
    """
    Intervention that replaces predicted concepts with ground truth values.

    Implements do(C=c_true) operations by mixing predicted and ground truth
    concept values based on a binary mask.

    Args:
        ground_truth: Ground truth concept values of shape (batch_size, n_concepts).
    """

    def __init__(self, constants: torch.Tensor | float):
        super().__init__()
        const = constants if torch.is_tensor(constants) else torch.tensor(constants)
        self.register_buffer("constants", const)

    def forward(self, x, *args, **kwargs):
        v = self.constants
        try:
            v = torch.broadcast_to(v, x.shape)
        except RuntimeError as e:
            raise ValueError(
                f"constants of shape {tuple(self.constants.shape)} cannot be "
                f"broadcast to concept tensor shape {tuple(x.shape)} "
                f"(expects scalar, [F], or any shape broadcastable against "
                f"[..., F])"
            ) from e

        return v.to(dtype=x.dtype, device=x.device)
