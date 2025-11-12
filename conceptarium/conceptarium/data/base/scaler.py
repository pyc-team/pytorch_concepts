from abc import ABC, abstractmethod
from torch import Tensor

class Scaler(ABC):
    """Abstract base class for data scaling transformations.
    
    Provides interface for fitting scalers to data and transforming/inverse-transforming
    tensors. Scalers can operate along specified dimensions of the input tensor.
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
