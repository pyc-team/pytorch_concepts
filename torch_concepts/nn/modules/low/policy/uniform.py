import torch

from ..base.layer import BaseConceptLayer


class UniformPolicy(BaseConceptLayer):
    """
    Uniform intervention policy that assigns equal priority to all concepts.

    This policy returns zeros for all concepts, indicating uniform/equal
    uncertainty or priority across all concepts. Useful as a baseline where
    no concept is preferred over others.

    Attributes:
        out_concepts (int): Number of output concepts.

    Args:
        out_concepts: Number of output concepts.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import UniformPolicy
        >>>
        >>> # Create uniform policy
        >>> policy = UniformPolicy(out_concepts=10)
        >>>
        >>> # Generate random concepts
        >>> concepts = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>>
        >>> # Apply policy - returns zeros (uniform priority)
        >>> scores = policy(concepts)
        >>> print(scores.shape)  # torch.Size([4, 10])
        >>> print(torch.all(scores == 0.0))  # True
        >>>
        >>> # Useful for baseline comparisons
        >>> # All concepts have equal intervention priority
        >>> print(scores.mean())  # tensor(0.)
        >>> print(scores.std())   # tensor(0.)
    """

    def __init__(
        self,
        out_concepts: int,
    ):
        super().__init__(
            out_concepts=out_concepts,
        )

    def forward(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate uniform (zero) intervention scores.

        Args:
            concepts: Input concepts of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Zeros tensor of same shape as input.
        """
        return torch.zeros_like(concepts)
