from typing import Optional

import torch

from ...concepts.annotations import Annotations
from ...nn.base.layer import BaseEncoder, BasePredictor

import inspect

def _filter_kwargs_for_ctor(cls, **kwargs):
    """Return only kwargs accepted by cls.__init__, skipping 'self'."""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # If the class accepts **kwargs, we can pass everything through.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = {
        name for name, p in params.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in allowed}

def instantiate_adaptive(module_cls, *args, drop_none=True, **kwargs):
    """Instantiate module_cls with only supported kwargs (optionally dropping None)."""
    if drop_none:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    filtered = _filter_kwargs_for_ctor(module_cls, **kwargs)
    return module_cls(*args, **filtered)



class Propagator(torch.nn.Module):
    def __init__(self,
                 module_cls: type[torch.nn.Module],  # Stores the class reference
                 *module_args,
                 **module_kwargs):
        super().__init__()

        # Store the module class and any additional keyword arguments
        self._module_cls = module_cls
        self._module_args = module_args
        self._module_kwargs = module_kwargs

        # The actual module is initially None.
        # It MUST be a torch.nn.Module or ModuleList/Sequential, not a lambda.
        self.module = None

    def build(self,
              out_annotations: Annotations,  # Assuming Annotations is a defined type
              in_features_logits: Optional[int],
              in_features_embedding: Optional[int],
              in_features_exogenous: Optional[int],
              ) -> torch.nn.Module:
        """
        Constructor method to instantiate the underlying module with required arguments.
        """
        # Instantiate the module using the stored class and kwargs
        # The module is instantiated with the provided arguments
        self.module = instantiate_adaptive(
            self._module_cls,
            *self._module_args,
            **{
                "in_features_logits": in_features_logits,
                "in_features_embedding": in_features_embedding,
                "in_features_exogenous": in_features_exogenous,
                "out_annotations": out_annotations,
                **self._module_kwargs,  # user-provided extras
            }
        )

        # Crucial for PyTorch: Check if the module is properly registered
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("The instantiated module is not a torch.nn.Module.")

        return self.module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass calls the instantiated module.
        """
        if self.module is None:
            raise RuntimeError(
                "Propagator module not built. Call .build(in_features, annotations) first."
            )

        # Forward calls the *instantiated* module instance
        return self.module(x, *args, **kwargs)
