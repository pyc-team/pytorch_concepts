"""
LazyConstructor module for delayed module instantiation.

This module provides a wrapper that delays the instantiation of neural network
modules until the required dimensions are known, enabling flexible model construction.
"""
from typing import Optional

import torch

import inspect

def _filter_kwargs_for_ctor(cls, **kwargs):
    """
    Return only kwargs accepted by cls.__init__, skipping 'self'.

    This helper function filters keyword arguments to only include those
    that are accepted by a class's constructor, preventing errors from
    passing unsupported arguments.

    Args:
        cls: The class to check constructor signature for.
        **kwargs: Keyword arguments to filter.

    Returns:
        dict: Filtered keyword arguments accepted by the class constructor.

    Example:
        >>> import torch.nn as nn
        >>> from torch_concepts.nn.modules.low.lazy import _filter_kwargs_for_ctor
        >>>
        >>> # Filter kwargs for Linear layer
        >>> kwargs = {'in_features': 10, 'out_features': 5, 'unknown_param': 42}
        >>> filtered = _filter_kwargs_for_ctor(nn.Linear, **kwargs)
        >>> print(filtered)
        {'in_features': 10, 'out_features': 5}
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # # If the class accepts **kwargs, we can pass everything through.
    # if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
    #     return kwargs

    allowed = {
        name for name, p in params.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in allowed}

def instantiate_adaptive(module_cls, *args, drop_none=True, **kwargs):
    """
    Instantiate module_cls with only supported kwargs (optionally dropping None).

    This function adaptively instantiates a module class by filtering the
    keyword arguments to only include those accepted by the class constructor.

    Args:
        module_cls: The module class to instantiate.
        *args: Positional arguments for the constructor.
        drop_none: If True, remove keyword arguments with None values (default: True).
        **kwargs: Keyword arguments for the constructor.

    Returns:
        An instance of module_cls.

    Example:
        >>> import torch.nn as nn
        >>> from torch_concepts.nn.modules.low.lazy import instantiate_adaptive
        >>>
        >>> # Instantiate a Linear layer with extra kwargs
        >>> kwargs = {'in_features': 10, 'out_features': 5, 'extra': None}
        >>> layer = instantiate_adaptive(nn.Linear, **kwargs)
        >>> print(layer)
        Linear(in_features=10, out_features=5, bias=True)
    """
    if drop_none:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    filtered = _filter_kwargs_for_ctor(module_cls, **kwargs)
    return module_cls(*args, **filtered)



class LazyConstructor(torch.nn.Module):
    """
    Delayed module instantiation wrapper for flexible neural network construction.

    The LazyConstructor class stores a module class and its initialization arguments,
    delaying actual instantiation until the required feature dimensions are known.
    This enables building models where concept dimensions are determined dynamically.

    Attributes:
        module (torch.nn.Module): The instantiated module (None until build() is called).

    Args:
        module_cls: The class of the module to instantiate.
        *module_args: Positional arguments for module instantiation.
        **module_kwargs: Keyword arguments for module instantiation.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LazyConstructor
        >>> from torch_concepts.nn import LinearCC
        >>>
        >>> # Create a propagator for a predictor
        >>> lazy_constructor = LazyConstructor(
        ...     LinearCC,
        ...     activation=torch.sigmoid
        ... )
        >>>
        >>> # Build the module when dimensions are known
        >>> module = lazy_constructor.build(
        ...     out_features=3,
        ...     in_features_endogenous=5,
        ...     in_features=None,
        ...     in_features_exogenous=None
        ... )
        >>>
        >>> # Use the module
        >>> x = torch.randn(2, 5)
        >>> output = lazy_constructor(x)
        >>> print(output.shape)
        torch.Size([2, 3])
    """
    def __init__(self,
                 module_cls: type[torch.nn.Module],  # Stores the class reference
                 *module_args,
                 **module_kwargs):
        """
        Initialize the LazyConstructor with a module class and its arguments.

        Args:
            module_cls: The class of the module to instantiate later.
            *module_args: Positional arguments for module instantiation.
            **module_kwargs: Keyword arguments for module instantiation.
        """
        super().__init__()

        # Store the module class and any additional keyword arguments
        self._module_cls = module_cls
        self._module_args = module_args
        self._module_kwargs = module_kwargs

        # The actual module is initially None.
        # It MUST be a torch.nn.Module or ModuleList/Sequential, not a lambda.
        self.module = None

    def build(self,
              out_features: int,
              in_features_endogenous: Optional[int],
              in_features: Optional[int],
              in_features_exogenous: Optional[int],
              **kwargs
              ) -> torch.nn.Module:
        """
        Build and instantiate the underlying module with required arguments.

        This method instantiates the stored module class with the provided
        feature dimensions and any additional arguments.

        Args:
            out_features: Number of output features.
            in_features_endogenous: Number of input logit features (optional).
            in_features: Number of input latent features (optional).
            in_features_exogenous: Number of exogenous input features (optional).
            **kwargs: Additional keyword arguments for the module.

        Returns:
            torch.nn.Module: The instantiated module.

        Raises:
            TypeError: If the instantiated object is not a torch.nn.Module.

        Example:
            >>> import torch
            >>> from torch_concepts.nn import LazyConstructor
            >>> from torch_concepts.nn import LinearCC
            >>>
            >>> lazy_constructor = LazyConstructor(LinearCC)
            >>> module = lazy_constructor.build(
            ...     out_features=3,
            ...     in_features_endogenous=5,
            ...     in_features=None,
            ...     in_features_exogenous=None
            ... )
            >>> print(type(module).__name__)
            LinearCC
        """
        # Instantiate the module using the stored class and kwargs
        # The module is instantiated with the provided arguments
        self.module = instantiate_adaptive(
            self._module_cls,
            *self._module_args,
            **{
                "in_features": in_features,
                "in_features_endogenous": in_features_endogenous,
                "in_features_exogenous": in_features_exogenous,
                "out_features": out_features,
                **self._module_kwargs,  # user-provided extras
                **kwargs,  # additional kwargs if provided
            }
        )

        # Crucial for PyTorch: Check if the module is properly registered
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("The instantiated module is not a torch.nn.Module.")

        return self.module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the instantiated module.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the module.
            **kwargs: Additional keyword arguments for the module.

        Returns:
            torch.Tensor: Output from the module.

        Raises:
            RuntimeError: If the module has not been built yet.

        Example:
            >>> import torch
            >>> from torch_concepts.nn import LazyConstructor
            >>> from torch_concepts.nn import LinearCC
            >>>
            >>> # Create and build propagator
            >>> lazy_constructor = LazyConstructor(LinearCC)
            >>> lazy_constructor.build(
            ...     out_features=3,
            ...     in_features_endogenous=5,
            ...     in_features=None,
            ...     in_features_exogenous=None
            ... )
            >>>
            >>> # Forward pass
            >>> x = torch.randn(2, 5)
            >>> output = lazy_constructor(x)
            >>> print(output.shape)
            torch.Size([2, 3])
        """
        if self.module is None:
            raise RuntimeError(
                "LazyConstructor module not built. Call .build(in_features, annotations) first."
            )

        # Forward calls the *instantiated* module instance
        return self.module(x, *args, **kwargs)
