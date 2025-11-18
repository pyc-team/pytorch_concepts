"""Standard scaling (z-score normalization) for data preprocessing.

This module provides StandardScaler for normalizing data to zero mean and
unit variance, similar to scikit-learn's StandardScaler but for PyTorch tensors.
"""

from typing import Tuple, Union
import torch
from torch import Tensor

from ..base.scaler import Scaler

def zeros_to_one_(scale: Union[float, Tensor]) -> Union[float, Tensor]:
    """Set to 1 scales of near-constant features to avoid division by zero.
    
    Detects features with near-zero variance (within machine precision) and
    sets their scale to 1.0 to prevent numerical instability. Operates in-place
    for tensor inputs.
    
    Adapted from sklearn.preprocessing._data._handle_zeros_in_scale and
    tsl.data.preprocessing.scalers.zeros_to_one_
    
    Args:
        scale (Union[float, Tensor]): Scalar or tensor of scale values to check.
        
    Returns:
        Union[float, Tensor]: Modified scale with near-zero values replaced by 1.0.
        
    Example:
        >>> scales = torch.tensor([1.0, 0.0000001, 2.5, 0.0])
        >>> zeros_to_one_(scales)
        tensor([1.0000, 1.0000, 2.5000, 1.0000])
    """
    if isinstance(scale, (int, float)):
        return 1.0 if torch.isclose(torch.tensor(scale), torch.tensor(0.0)).item() else scale
    
    eps = 10 * torch.finfo(scale.dtype).eps
    zeros = torch.isclose(scale, torch.tensor(0.0, device=scale.device, dtype=scale.dtype), atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


class StandardScaler(Scaler):
    """Z-score normalization scaler for PyTorch tensors.
    
    Standardizes features by removing the mean and scaling to unit variance:
        z = (x - μ) / σ
    
    This scaler is useful for:
    - Normalizing input features before training
    - Ensuring all features are on the same scale
    - Improving gradient flow and training stability
    
    Args:
        axis (Union[int, Tuple], optional): Axis or axes along which to compute
            mean and standard deviation. Typically 0 (across samples) for
            feature-wise normalization. Defaults to 0.
            
    Attributes:
        mean (Tensor): Computed mean value(s) from fitted data.
        std (Tensor): Computed standard deviation(s) from fitted data.
        
    Example:
        >>> # Normalize a batch of features
        >>> scaler = StandardScaler(axis=0)
        >>> X_train = torch.randn(1000, 50)  # 1000 samples, 50 features
        >>> X_train_scaled = scaler.fit_transform(X_train)
        >>> 
        >>> # Transform test data using training statistics
        >>> X_test = torch.randn(200, 50)
        >>> X_test_scaled = scaler.transform(X_test)
        >>> 
        >>> # Inverse transform to original scale
        >>> X_recovered = scaler.inverse_transform(X_test_scaled)
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
