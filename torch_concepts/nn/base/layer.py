from typing import Union, Dict, Tuple

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
        out_annotations: Annotations,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.out_annotations = out_annotations

        self.concept_axis = 1
        self.out_probs_dim = out_annotations.shape[1]

    @property
    def in_concept_features(self) -> Dict[str, int]:
        in_concept_features = {}
        for key, shape in self.in_concept_shapes.items():
            in_concept_features[key] = np.prod(shape).item()
        return in_concept_features

    @property
    def out_concept_features(self) -> Dict[str, int]:
        out_concept_features = {}
        for key, shape in self.out_concept_shapes.items():
            out_concept_features[key] = np.prod(shape).item()
        return out_concept_features

    @property
    @abstractmethod
    def in_concept_shapes(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def out_concept_shapes(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def in_concepts(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def out_concepts(self) -> Tuple[str, ...]:
        raise NotImplementedError

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
                annotations=self.out_annotations
            )


class BaseEncoder(BaseConceptLayer):
    """
    BaseConceptLayer is an abstract base class for concept encoder layers.
    The output objects are ConceptTensors.
    """
    def __init__(self, in_features: int, out_annotations: Annotations, *args, **kwargs):
        super().__init__(
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )
        self._in_features = in_features
        in_concept_shapes = self.in_concept_shapes
        in_concept_features = self.in_concept_features

    @property
    def in_concept_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"residual": (self._in_features,)}

    @property
    def in_concepts(self) -> Tuple[str]:
        return ("residual",)

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


class BasePredictor(BaseConceptLayer):
    """
    BasePredictor is an abstract base class for concept predictor layers.
    The input objects are ConceptTensors and the output objects are ConceptTensors with concept probabilities only.
    """
    def __init__(self, in_concept_features: Union[Tuple[Dict[str, int]], Dict[str, int]], out_annotations: Annotations, *args, **kwargs):
        super().__init__(
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )
        self._in_concept_features = in_concept_features
        in_concept_shapes = self.in_concept_shapes
        in_concept_features = self.in_concept_features
        out_concept_shapes = self.out_concept_shapes
        out_concept_features = self.out_concept_features

    @property
    def out_concept_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"concept_probs": (self.out_probs_dim,)}

    @property
    def out_concepts(self) -> Tuple[str]:
        return ("concept_probs",)

    def forward(
        self,
        x: ConceptTensor,
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
        if not isinstance(x, ConceptTensor):
            raise TypeError(
                f"The input to {self.__class__.__name__}.forward() must be a ConceptTensor, "
                f"but got {type(x)} instead."
            )

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
        x: ConceptTensor,
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
