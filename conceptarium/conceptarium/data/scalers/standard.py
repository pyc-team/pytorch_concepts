from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
from torch import Tensor

from ..base.scaler import Scaler

def zeros_to_one_(scale: Union[float, Tensor]) -> Union[float, Tensor]:
    """Set to 1 scales of near constant features, detected by identifying
    scales close to machine precision, in place.
    Adapted from :class:`sklearn.preprocessing._data._handle_zeros_in_scale`
        and from: `tsl.data.preprocessing.scalers.zeros_to_one_`

    Args:
        scale: Scalar or tensor of scale values to check and modify.
        
    Returns:
        Modified scale with near-zero values replaced by 1.0.
    """
    if isinstance(scale, (int, float)):
        return 1.0 if torch.isclose(torch.tensor(scale), torch.tensor(0.0)).item() else scale
    
    eps = 10 * torch.finfo(scale.dtype).eps
    zeros = torch.isclose(scale, torch.tensor(0.0, device=scale.device, dtype=scale.dtype), atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


class StandardScaler(Scaler):
    """Z-score normalization scaler.
    Standardizes features by removing the mean and scaling to unit variance:
        z = (x - μ) / σ
    Attributes:
        mean: Mean value(s) computed from fitted data.
        std: Standard deviation(s) computed from fitted data.
    """

    def __init__(self, axis: Union[int, Tuple] = 0):
        """Initialize the StandardScaler.
        Args:
            axis: Axis or axes along which to compute statistics (default: 0).
        """
        super(StandardScaler, self).__init__()
        self.axis = axis

    def fit(self, x: Tensor) -> "StandardScaler":
        """Compute mean and standard deviation along specified dimension.
        Args:
            x: Input tensor to compute statistics from.
        Returns:
            self: The fitted scaler instance for method chaining.
        """
        self.mean = x.mean(dim=self.axis, keepdim=True)
        self.std = x.std(dim=self.axis, keepdim=True)

        self.std = zeros_to_one_(self.std)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Standardize the input tensor using fitted statistics.
        Args:
            x: Input tensor to standardize.
        Returns:
            Standardized tensor with zero mean and unit variance.
        """
        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the standardization to recover original scale.
        Args:
            x: Standardized tensor to inverse-transform.
        Returns:
            Tensor in original scale.
        """
        return x * self.std + self.mean
