from typing import Union

import numpy as np
import torch

from abc import ABC, abstractmethod
from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    """

    def __init__(
        self,
        in_features: int,
        annotations: Annotations,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.annotations = annotations

        self.concept_axis = 1
        self._shape = annotations.shape[1:]
        self.output_size = np.prod(self._shape).item()

    def shape(self):
        return self._shape

    def annotate(
        self,
        x: torch.Tensor,
    ) -> AnnotatedTensor:
        """
        Annotate tensor.

        Args:
            x (torch.Tensor): A tensor compatible with the layer's annotations.

        Returns:
            AnnotatedTensor: Annotated tensor.
        """
        return AnnotatedTensor(
            data=x,
            annotations=self.annotations
        )


class BaseEncoderLayer(BaseConceptLayer):
    """
    BaseConceptLayer is an abstract base class for concept encoder layers.
    The output objects are ConceptTensors.
    """
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Forward pass of a ConceptLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Predicted concept object.
        """
        # 1. Call the subclass's logic
        output: ConceptTensor = self.encode(x, *args, **kwargs)

        # 2. **RUNTIME CHECK:**
        # Enforce the output is a ConceptTensor
        if not isinstance(output, ConceptTensor):
            # Raise an error if the contract is violated
            raise TypeError(
                f"The output of {self.__class__.__name__}.forward() must be a ConceptTensor, "
                f"but got {type(output)} instead."
            )
        # Enforce at least one of concept_probs, concept_embs, residual is not None
        if output.concept_probs is None and output.concept_embs is None and output.residual is None:
            # Raise an error if the contract is violated
            raise ValueError(
                f"The output of {self.__class__.__name__}.forward() must be a ConceptTensor with "
                f"at least one of 'concept_probs', 'concept_embs', or 'residual' defined."
            )
        return output

    @abstractmethod
    def encode(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Encode input tensor to ConceptTensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Encoded concept object.
        """
        raise NotImplementedError("encode")


class BasePredictorLayer(BaseConceptLayer):
    """
    BasePredictorLayer is an abstract base class for concept predictor layers.
    The input objects are ConceptTensors and the output objects are ConceptTensors with concept probabilities only.
    """
    def forward(
        self,
        x: Union[torch.Tensor, ConceptTensor],
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Forward pass of a ConceptLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Predicted concept object.
        """
        # 1. Call the subclass's logic
        output: ConceptTensor = self.predict(x, *args, **kwargs)

        # 2. **RUNTIME CHECK:** Enforce concept_probs is not None
        if output.concept_probs is None:
            # Raise an error if the contract is violated
            raise ValueError(
                f"The output of {self.__class__.__name__}.forward() must have "
                f"'concept_probs' not set to None."
            )
        return output

    @abstractmethod
    def predict(
        self,
        x: Union[torch.Tensor, ConceptTensor],
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Predict concept probabilities from input tensor or ConceptTensor.

        Args:
            x (Union[torch.Tensor, ConceptTensor]): Input tensor or ConceptTensor.
        Returns:
            ConceptTensor: Predicted concept object with concept probabilities.
        """
        raise NotImplementedError("predict")
