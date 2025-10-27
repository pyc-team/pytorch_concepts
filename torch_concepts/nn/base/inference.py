import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple, Union, Callable, Optional

import fnmatch
import torch
import torch.nn as nn
import torch.distributions as D

from torch_concepts import ConceptTensor


class BaseInference(torch.nn.Module):
    """
    BaseInference is an abstract class for inference modules.
    """
    def __init__(self, model: torch.nn.Module):
        super(BaseInference, self).__init__()
        self.model = model

    def forward(self,
                x: torch.Tensor,
                *args,
                **kwargs) -> ConceptTensor:
        return self.query(x, *args, **kwargs)

    @abstractmethod
    def query(self,
              x: torch.Tensor,
              c: torch.Tensor,
              *args,
              **kwargs) -> ConceptTensor:
        """
        Query model to get concepts.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Concept tensor for interventions. Defaults to None.

        Returns:
            ConceptTensor: Queried concepts.
        """
        raise NotImplementedError


class BaseIntervention(BaseInference, ABC):
    """
    Base class for interventions. Subclass and implement `forward`.
    """
    def __init__(self, module_dict: torch.nn.ModuleDict, *args, **kwargs):
        super().__init__(model=module_dict)
        self.out_features = None

    def forward(self, key: str, *args, **kwargs) -> ConceptTensor:
        """
        Apply intervention to the module identified by `key`.
        """
        if key not in self.model:
            raise KeyError(f"ModuleDict has no key '{key}'")

        self.out_features = self.model[key].out_features
        return self.query(self.model[key], *args, **kwargs)
