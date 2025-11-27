"""Abstract base class for data scaling transformations.

This module defines the Scaler interface that all data scalers must implement.
Scalers are used to normalize and denormalize data during training and inference.
"""

from abc import ABC, abstractmethod
from torch import Tensor

class Scaler(ABC):
    """Abstract base class for data scaling transformations.
    
    Provides a consistent interface for fitting scalers to data and applying
    forward/inverse transformations. All concrete scaler implementations should
    inherit from this class and implement fit(), transform(), and
    inverse_transform() methods.
    
    Args:
        bias (float, optional): Initial bias value. Defaults to 0.0.
        scale (float, optional): Initial scale value. Defaults to 1.0.
        
    Example:
        >>> class MinMaxScaler(Scaler):
        ...     def fit(self, x, dim=0):
        ...         self.min = x.min(dim=dim, keepdim=True)[0]
        ...         self.max = x.max(dim=dim, keepdim=True)[0]
        ...         return self
        ...     
        ...     def transform(self, x):
        ...         return (x - self.min) / (self.max - self.min)
        ...     
        ...     def inverse_transform(self, x):
        ...         return x * (self.max - self.min) + self.min
    """

    def __init__(self, bias=0., scale=1.):
        self.bias = bias
        self.scale = scale
        super(Scaler, self).__init__()

    @abstractmethod
    def fit(self, x: Tensor, dim: int = 0) -> "Scaler":
        """Fit the scaler to the input data.
        Args:
            x: Input tensor to fit the scaler to.
            dim: Dimension along which to compute statistics (default: 0).
        Returns:
            self: The fitted scaler instance for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """Apply the fitted transformation to the input tensor.
        Args:
            x: Input tensor to transform.
        Returns:
            Transformed tensor with same shape as input.
        """
        pass

    @abstractmethod
    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the transformation to recover original data.
        Args:
            x: Transformed tensor to inverse-transform.
        Returns:
            Tensor in original scale with same shape as input.
        """
        pass

    def fit_transform(self, x: Tensor, dim: int = 0) -> Tensor:
        """Fit the scaler and transform the input data in one operation.
        Args:
            x: Input tensor to fit and transform.
            dim: Dimension along which to compute statistics (default: 0).
        Returns:
            Transformed tensor with same shape as input.
        """
        self.fit(x, dim=dim)
        return self.transform(x)
