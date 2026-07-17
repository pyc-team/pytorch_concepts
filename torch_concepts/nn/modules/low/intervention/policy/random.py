import torch

from ...base.intervention import BaseInterventionPolicy


class RandomPolicy(BaseInterventionPolicy):
    """
    Random intervention policy that generates random values for concept selection.

    This policy generates random values scaled by a factor, useful for random
    baseline comparisons in intervention experiments.

    Attributes:
        scale (float): Scaling factor for random values.

    Args:
        scale: Scaling factor for random values (default: 1.0).
    """

    def __init__(
        self,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        concepts: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate random intervention scores.

        Args:
            concepts: Input concepts of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Random scores of same shape as input, scaled by self.scale.
        """
        return torch.rand_like(concepts).abs() * self.scale
