import torch

from ..base.layer import BaseConceptLayer


class RandomPolicy(BaseConceptLayer):
    """
    Random intervention policy that generates random values for concept selection.

    This policy generates random values scaled by a factor, useful for random
    baseline comparisons in intervention experiments.

    Attributes:
        out_concepts (int): Number of output concepts.
        scale (float): Scaling factor for random values.

    Args:
        out_concepts: Number of output concepts.
        scale: Scaling factor for random values (default: 1.0).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import RandomPolicy
        >>>
        >>> # Create random policy
        >>> policy = RandomPolicy(out_concepts=10, scale=2.0)
        >>>
        >>> # Generate random concepts
        >>> concepts = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>>
        >>> # Apply policy to get random intervention scores
        >>> scores = policy(concepts)
        >>> print(scores.shape)  # torch.Size([4, 10])
        >>> print(scores.min() >= 0.0)  # True (absolute values)
        >>> print(scores.max() <= 2.0)  # True (scaled by 2.0)
        >>>
        >>> # Each call generates different random values
        >>> scores2 = policy(concepts)
        >>> print(torch.equal(scores, scores2))  # False
    """

    def __init__(
        self,
        out_concepts: int,
        scale: float = 1.0,
    ):
        super().__init__(
            out_concepts=out_concepts,
        )
        self.scale = scale

    def forward(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate random intervention scores.

        Args:
            concepts: Input concepts of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Random scores of same shape as input, scaled by self.scale.
        """
        return torch.rand_like(concepts).abs() * self.scale
