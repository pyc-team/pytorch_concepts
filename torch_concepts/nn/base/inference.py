"""
Base inference and intervention classes for concept-based models.

This module provides abstract base classes for implementing inference mechanisms
and intervention strategies in concept-based models.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseInference(torch.nn.Module):
    """
    Abstract base class for inference modules.

    Inference modules define how to query concept-based models to obtain
    concept predictions, supporting various inference strategies such as
    forward inference, ancestral sampling, or stochastic inference.

    Example:
        >>> class MyInference(BaseInference):
        ...     def query(self, x):
        ...         # Custom inference logic
        ...         return concepts
    """
    def __init__(self):
        """Initialize the inference module."""
        super(BaseInference, self).__init__()

    def forward(self,
                x: torch.Tensor,
                *args,
                **kwargs) -> torch.Tensor:
        """
        Forward pass delegates to the query method.

        Args:
            x: Input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Queried concepts.
        """
        return self.query(x, *args, **kwargs)

    @abstractmethod
    def query(self,
              *args,
              **kwargs) -> torch.Tensor:
        """
        Query model to get concepts.

        This method must be implemented by subclasses to define the
        specific inference strategy.

        Args:
            *args: Variable length argument list (typically includes input x).
            **kwargs: Arbitrary keyword arguments (may include intervention c).

        Returns:
            torch.Tensor: Queried concept predictions.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError


class BaseIntervention(BaseInference, ABC):
    """
    Abstract base class for intervention modules.

    Intervention modules modify concept-based models by replacing certain
    modules, enabling causal reasoning and what-if analysis.

    This class provides a framework for implementing different intervention
    strategies on concept-based models.

    Attributes:
        model (nn.Module): The concept-based model to apply interventions to.

    Args:
        model: The neural network model to intervene on.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the intervention module.

        Args:
            model (nn.Module): The concept-based model to apply interventions to.
        """
        super().__init__()
        self.model = model
