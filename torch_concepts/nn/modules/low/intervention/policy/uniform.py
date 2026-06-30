import torch

from ...base.intervention import BaseInterventionPolicy


class UniformPolicy(BaseInterventionPolicy):
    """
    Uniform intervention policy that assigns equal priority to all concepts.

    This policy returns zeros for all concepts, indicating uniform/equal
    uncertainty or priority across all concepts. Useful as a baseline where
    no concept is preferred over others.

    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        concepts: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate uniform (zero) intervention scores.

        Args:
            concepts: Input concepts of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Zeros tensor of same shape as input.
        """
        return torch.zeros_like(concepts)
