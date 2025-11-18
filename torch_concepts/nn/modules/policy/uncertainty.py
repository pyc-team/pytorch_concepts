import torch

from ....nn.base.layer import BaseConceptLayer


class UncertaintyInterventionPolicy(BaseConceptLayer):
    """
    Uncertainty-based intervention policy using concept logit magnitudes.

    This policy uses the absolute value of concept logits as a measure of
    certainty/uncertainty. Higher absolute values indicate higher certainty,
    while values near zero indicate higher uncertainty.

    Attributes:
        out_features (int): Number of output features.

    Args:
        out_features: Number of output concept features.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import UncertaintyInterventionPolicy
        >>>
        >>> # Create uncertainty policy
        >>> policy = UncertaintyInterventionPolicy(out_features=10)
        >>>
        >>> # Generate concept logits with varying confidence
        >>> logits = torch.tensor([
        ...     [3.0, -2.5, 0.1, -0.2, 4.0],  # High confidence for 1st, 2nd, 5th
        ...     [0.5, 0.3, -0.4, 2.0, -1.5]   # Mixed confidence
        ... ])
        >>>
        >>> # Apply policy - returns absolute values (certainty scores)
        >>> scores = policy(logits)
        >>> print(scores)
        >>> # tensor([[3.0, 2.5, 0.1, 0.2, 4.0],
        >>> #         [0.5, 0.3, 0.4, 2.0, 1.5]])
        >>>
        >>> # Higher scores = higher certainty = lower intervention priority
        >>> # For intervention, you'd typically intervene on LOW scores
        >>> print(scores[0].argmin())  # tensor(2) - most uncertain concept
        >>> print(scores[0].argmax())  # tensor(4) - most certain concept
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
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute certainty scores from concept logits.

        Args:
            logits: Input concept logits of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Absolute values (certainty scores) of same shape as input.
        """
        return logits.abs()
