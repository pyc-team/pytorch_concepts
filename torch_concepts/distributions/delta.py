"""
Delta (deterministic) distribution implementation.

This module provides a deterministic distribution that always returns a fixed value,
useful for representing deterministic concepts in probabilistic models.
"""
import torch
from torch.distributions import Distribution
from typing import List, Dict, Any, Union, Optional


class Delta(Distribution):
    """
    Delta (Dirac delta) distribution - a deterministic distribution.

    This distribution always returns the same fixed value when sampled,
    making it useful for representing deterministic variables in
    Probabilistic Models.

    The Delta distribution has zero variance and assigns all probability
    mass to a single point.

    Attributes:
        arg_constraints (Dict): Empty dict - no constraints on parameters.
        support (Optional[torch.Tensor]): Support of the distribution (None for Delta).
        has_rsample (bool): Whether reparameterized sampling is supported (False).

    Args:
        value: The deterministic value (list or tensor).
        validate_args: Whether to validate arguments (default: None).

    Properties:
        mean: Returns the deterministic value.

    Examples:
        >>> import torch
        >>> from torch_concepts.distributions import Delta
        >>> dist = Delta(torch.tensor([1.0, 2.0, 3.0]))
        >>> sample = dist.sample()
        >>> print(sample)  # tensor([1., 2., 3.])
        >>> print(dist.mean)  # tensor([1., 2., 3.])
    """
    arg_constraints: Dict[str, Any] = {}
    support: Optional[torch.Tensor] = None
    has_rsample = False

    def __init__(self, value: Union[List[float], torch.Tensor], validate_args=None):
        """
        Initialize a Delta distribution.

        Args:
            value: The fixed value this distribution returns (list or tensor).
            validate_args: Whether to validate arguments (default: None).
        """
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)

        super().__init__(batch_shape=torch.Size([]), validate_args=validate_args)
        self._value = value.clone()

    @property
    def mean(self):
        """
        Return the mean of the distribution.

        For a Delta distribution, the mean is the deterministic value itself.

        Returns:
            torch.Tensor: The deterministic value.
        """
        return self._value

    def sample(self, sample_shape=torch.Size()):
        """
        Generate a sample from the distribution.

        For a Delta distribution, always returns the deterministic value.

        Args:
            sample_shape: Shape of the sample (default: empty tuple).

        Returns:
            torch.Tensor: The deterministic value.
        """
        return self._value

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate a reparameterized sample from the distribution.

        For a Delta distribution, this is the same as sample().

        Args:
            sample_shape: Shape of the sample (default: empty tuple).

        Returns:
            torch.Tensor: The deterministic value.
        """
        return self._value

    def log_prob(self, value):
        """
        Calculate the log probability of a value.

        For a Delta distribution, technically the log probability is
        -inf for any value except the deterministic value, and +inf
        at the deterministic value. This implementation returns 0.

        Args:
            value: Value to compute log probability for.

        Returns:
            torch.Tensor: Log probability (zeros).
        """
        return torch.zeros(value.shape[:-len(self.event_shape)])

    def __repr__(self):
        """
        Return string representation of the distribution.

        Returns:
            str: String representation showing the value shape.
        """
        return f"Delta(value_shape={self._value.shape})"
