import torch

from ..base.layer import BaseConceptLayer


class UniformPolicy(BaseConceptLayer):
    """
    Uniform intervention policy that assigns equal priority to all concepts.

    This policy returns zeros for all concepts, indicating uniform/equal
    uncertainty or priority across all concepts. Useful as a baseline where
    no concept is preferred over others.

    Attributes:
        out_features (int): Number of output features.

    Args:
        out_features: Number of output concept features.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import UniformPolicy
        >>>
        >>> # Create uniform policy
        >>> policy = UniformPolicy(out_features=10)
        >>>
        >>> # Generate random concept endogenous
        >>> endogenous = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>>
        >>> # Apply policy - returns zeros (uniform priority)
        >>> scores = policy(endogenous)
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
        out_features: int,
    ):
        super().__init__(
            out_features=out_features,
        )

    def forward(
        self,
        endogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate uniform (zero) intervention scores.

        Args:
            endogenous: Input concept endogenous of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Zeros tensor of same shape as input.
        """
        return torch.zeros_like(endogenous)
