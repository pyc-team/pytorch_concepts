import torch

from ..base.layer import BaseConceptLayer


class UncertaintyInterventionPolicy(BaseConceptLayer):
    """
    Uncertainty-based intervention policy using distance from a maximum uncertainty point.

    This policy measures uncertainty as the distance of concept endogenous from a
    maximum uncertainty point. Values closer to this point are considered more uncertain,
    while values further from this point are considered more certain.

    Attributes:
        out_features (int): Number of output features.
        max_uncertainty_point (float): The point where uncertainty is maximum.

    Args:
        out_features: Number of output concept features.
        max_uncertainty_point: The value representing maximum uncertainty (default: 0.0).
            Values closer to this point are more uncertain, values further away are more certain.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import UncertaintyInterventionPolicy
        >>>
        >>> # Create uncertainty policy with default max uncertainty point (0.0)
        >>> policy = UncertaintyInterventionPolicy(out_features=10)
        >>>
        >>> # Generate concept endogenous with varying confidence
        >>> endogenous = torch.tensor([
        ...     [3.0, -2.5, 0.1, -0.2, 4.0],  # High confidence for 1st, 2nd, 5th
        ...     [0.5, 0.3, -0.4, 2.0, -1.5]   # Mixed confidence
        ... ])
        >>>
        >>> # Apply policy - returns distance from max uncertainty point (certainty scores)
        >>> scores = policy(endogenous)
        >>> print(scores)
        >>> # tensor([[3.0, 2.5, 0.1, 0.2, 4.0],
        >>> #         [0.5, 0.3, 0.4, 2.0, 1.5]])
        >>>
        >>> # Higher scores = higher certainty = lower intervention priority
        >>> # For intervention, you'd typically intervene on LOW scores
        >>> print(scores[0].argmin())  # tensor(2) - most uncertain concept
        >>> print(scores[0].argmax())  # tensor(4) - most certain concept
        >>>
        >>> # Use custom max uncertainty point (e.g., 0.5 for probabilities)
        >>> policy_prob = UncertaintyInterventionPolicy(out_features=5, max_uncertainty_point=0.5)
        >>> probs = torch.tensor([[0.1, 0.5, 0.9, 0.45, 0.55]])
        >>> certainty = policy_prob(probs)
        >>> print(certainty)
        >>> # tensor([[0.4, 0.0, 0.4, 0.05, 0.05]])
        >>> # Values at 0.5 are most uncertain, values at 0.1 or 0.9 are most certain
    """

    def __init__(
        self,
        out_features: int,
        max_uncertainty_point: float = 0.0,
    ):
        super().__init__(
            out_features=out_features,
        )
        self.max_uncertainty_point = max_uncertainty_point

    def forward(
        self,
        endogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute certainty scores as distance from maximum uncertainty point.

        Args:
            endogenous: Input concept endogenous of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Distance from max uncertainty point (certainty scores) of same shape as input.
                Higher values indicate higher certainty (further from max uncertainty point).
                Lower values indicate higher uncertainty (closer to max uncertainty point).
        """
        return (endogenous - self.max_uncertainty_point).abs()
