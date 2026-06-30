import torch

from ...base.intervention import BaseInterventionPolicy


class UncertaintyInterventionPolicy(BaseInterventionPolicy):
    """
    Uncertainty-based intervention policy using distance from a maximum uncertainty point.

    This policy measures uncertainty as the distance of concepts from a
    maximum uncertainty point. Values closer to this point are considered more uncertain,
    while values further from this point are considered more certain.

    Attributes:
        max_uncertainty_point (float): The point where uncertainty is maximum.

    Args:
        max_uncertainty_point: The value representing maximum uncertainty (default: 0.0).
            Values closer to this point are more uncertain, values further away are more certain.
    """

    def __init__(
        self,
        max_uncertainty_point: float = 0.0,
    ):
        super().__init__()
        self.max_uncertainty_point = max_uncertainty_point

    def forward(
        self,
        concepts: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute certainty scores as distance from maximum uncertainty point.

        Args:
            concepts: Input concepts of shape (batch_size, n_concepts).

        Returns:
            torch.Tensor: Distance from max uncertainty point (certainty scores) of same shape as input.
                Higher values indicate higher certainty (further from max uncertainty point).
                Lower values indicate higher uncertainty (closer to max uncertainty point).
        """
        return (concepts - self.max_uncertainty_point).abs()
