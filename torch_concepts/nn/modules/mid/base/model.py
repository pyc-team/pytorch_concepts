"""
Base model class for concept-based architectures.

This module provides the abstract base class for all concept-based models,
defining the structure for models that use concept representations.
"""
from typing import Union

import torch
from torch.nn import Module

from .....annotations import Annotations
from ...low.lazy import LazyConstructor


class BaseConstructor(torch.nn.Module):
    """
    Abstract base class for all concept-based models.

    This class provides the foundation for building concept-based neural networks.

    Attributes:
        input_size (int): Size of the input features.
        annotations (Annotations): Concept annotations with metadata.
        labels (List[str]): List of concept labels.
        name2id (Dict[str, int]): Mapping from concept names to indices.

    Args:
        input_size: Size of the input features.
        annotations: Annotations object containing concept metadata.
        encoder: LazyConstructor layer for encoding root concepts from inputs.
        predictor: LazyConstructor layer for making predictions from concepts.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations, AxisAnnotation
        >>> from torch_concepts.nn import LazyConstructor
        >>> from torch_concepts.nn.modules.mid.base.model import BaseConstructor
        >>> from torch.distributions import RelaxedBernoulli
        >>>
        >>> # Create annotations for concepts
        >>> concept_labels = ('color', 'shape', 'size')
        >>> cardinalities = [1, 1, 1]
        >>> metadata = {
        ...     'color': {'distribution': RelaxedBernoulli},
        ...     'shape': {'distribution': RelaxedBernoulli},
        ...     'size': {'distribution': RelaxedBernoulli}
        ... }
        >>> annotations = Annotations({1: AxisAnnotation(
        ...     labels=concept_labels,
        ...     cardinalities=cardinalities,
        ...     metadata=metadata
        ... )})
        >>>
        >>> # Create a concrete model class
        >>> class MyConceptModel(BaseConstructor):
        ...     def __init__(self, input_size, annotations, encoder, predictor):
        ...         super().__init__(input_size, annotations, encoder, predictor)
        ...         # Build encoder and predictor
        ...         self.encoder = self._encoder_builder
        ...         self.predictor = self._predictor_builder
        ...
        ...     def forward(self, x):
        ...         concepts = self.encoder(x)
        ...         predictions = self.predictor(concepts)
        ...         return predictions
        >>>
        >>> # Create encoder and predictor propagators
        >>> encoder = torch.nn.Linear(784, 3)  # Simple encoder
        >>> predictor = torch.nn.Linear(3, 10)  # Simple predictor
        >>>
        >>> # Instantiate model
        >>> model = MyConceptModel(
        ...     input_size=784,
        ...     annotations=annotations,
        ...     encoder=encoder,
        ...     predictor=predictor
        ... )
        >>>
        >>> # Generate random input (e.g., flattened MNIST image)
        >>> x = torch.randn(8, 784)  # batch_size=8, pixels=784
        >>>
        >>> # Forward pass
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([8, 10])
        >>>
        >>> # Access concept labels
        >>> print(model.labels)  # ('color', 'shape', 'size')
        >>>
        >>> # Get concept index by name
        >>> idx = model.name2id['color']
        >>> print(idx)  # 0
    """

    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Union[LazyConstructor, Module],  # layer for root concepts
                 predictor: Union[LazyConstructor, Module],
                 *args,
                 **kwargs
    ):
        super(BaseConstructor, self).__init__()
        self.input_size = input_size
        self.annotations = annotations

        self._encoder_builder = encoder
        self._predictor_builder = predictor

        self.labels = annotations.get_axis_labels(axis=1)
        self.name2id = {name: i for i, name in enumerate(self.labels)}
