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
        >>> import torch
        >>> from torch_concepts.nn import BaseInference
        >>>
        >>> # Create a custom inference class
        >>> class SimpleInference(BaseInference):
        ...     def __init__(self, model):
        ...         super().__init__()
        ...         self.model = model
        ...
        ...     def query(self, x, **kwargs):
        ...         # Simple forward pass through model
        ...         return self.model(x)
        >>>
        >>> # Example usage
        >>> dummy_model = torch.nn.Linear(10, 5)
        >>> inference = SimpleInference(dummy_model)
        >>>
        >>> # Generate random input
        >>> x = torch.randn(2, 10)  # batch_size=2, input_features=10
        >>>
        >>> # Query concepts using forward method
        >>> concepts = inference(x)
        >>> print(concepts.shape)  # torch.Size([2, 5])
        >>>
        >>> # Or use query method directly
        >>> concepts = inference.query(x)
        >>> print(concepts.shape)  # torch.Size([2, 5])
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

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from torch_concepts.nn import BaseIntervention
        >>>
        >>> # Create a custom intervention class
        >>> class CustomIntervention(BaseIntervention):
        ...     def query(self, module_name, **kwargs):
        ...         # Get the module to intervene on
        ...         module = self.model.get_submodule(module_name)
        ...         # Apply intervention logic
        ...         return module(**kwargs)
        >>>
        >>> # Create a simple concept model
        >>> class ConceptModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.encoder = nn.Linear(10, 5)
        ...         self.predictor = nn.Linear(5, 3)
        ...
        ...     def forward(self, x):
        ...         concepts = torch.sigmoid(self.encoder(x))
        ...         return self.predictor(concepts)
        >>>
        >>> # Example usage
        >>> model = ConceptModel()
        >>> intervention = CustomIntervention(model)
        >>>
        >>> # Generate random input
        >>> x = torch.randn(2, 10)  # batch_size=2, input_features=10
        >>>
        >>> # Query encoder module
        >>> encoder_output = intervention.query('encoder', input=x)
        >>> print(encoder_output.shape)  # torch.Size([2, 5])
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the intervention module.

        Args:
            model (nn.Module): The concept-based model to apply interventions to.
        """
        super().__init__()
        self.model = model
