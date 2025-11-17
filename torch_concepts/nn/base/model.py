"""
Base model class for concept-based architectures.

This module provides the abstract base class for all concept-based models,
defining the structure for models that use concept representations.
"""
import torch

from torch_concepts import Annotations
from ..modules.propagator import Propagator


class BaseModel(torch.nn.Module):
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
        encoder: Propagator layer for encoding root concepts from inputs.
        predictor: Propagator layer for making predictions from concepts.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Example:
        >>> annotations = Annotations({1: AxisAnnotation(labels=['c1', 'c2', 'c3'])})
        >>> encoder = Propagator(...)
        >>> predictor = Propagator(...)
        >>> model = ConcreteModel(input_size=784, annotations=annotations,
        ...                       encoder=encoder, predictor=predictor)
    """

    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,  # layer for root concepts
                 predictor: Propagator,
                 *args,
                 **kwargs,
                 ):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.annotations = annotations

        self._encoder_builder = encoder
        self._predictor_builder = predictor

        self.labels = annotations.get_axis_labels(axis=1)
        self.name2id = {name: i for i, name in enumerate(self.labels)}
