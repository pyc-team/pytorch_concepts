from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


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
                **kwargs) -> torch.Tensor:
        return self.query(x, *args, **kwargs)

    @abstractmethod
    def query(self,
              *args,
              **kwargs) -> torch.Tensor:
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
    Returns {path: replacement_module}. For each path we compute the
    target feature shape (from the parent model or layer) and pass it
    into `query(..., target_shape=...)`.
    """
    def __init__(self, model: nn.Module):
        super().__init__(model=model)
