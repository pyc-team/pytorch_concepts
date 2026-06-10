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
        B, F = x.shape
        v = self.constants

        if v.dim() == 0:  # scalar
            v = v.view(1, 1).expand(B, F)
        elif v.dim() == 1:  # [F]
            assert v.numel() == F, f"constants [F] must have F={F}, got {v.numel()}"
            v = v.unsqueeze(0).expand(B, F)
        elif v.dim() == 2:
            b, f = v.shape
            assert f == F, f"constants second dim must be F={F}, got {f}"
            if b == 1:
                v = v.expand(B, F)  # [1, F] -> [B, F]
            else:
                assert b == B, f"constants first dim must be B={B} or 1, got {b}"
        else:
            raise ValueError("constants must be scalar, [F], [1, F], or [B, F]")

        return v.to(dtype=x.dtype, device=x.device)
