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
    def in_features(self) -> Dict[str, int]:
        in_features = {}
        for key, shape in self.in_shapes.items():
            in_features[key] = np.prod(shape).item()
        return in_features

    @property
    def out_features(self) -> Dict[str, int]:
        out_features = {}
        for key, shape in self.out_shapes.items():
            out_features[key] = np.prod(shape).item()
        return out_features

    @property
    def in_keys(self) -> Tuple[str, ...]:
        return tuple(self.in_shapes.keys())

    @property
    def out_keys(self) -> Tuple[str, ...]:
        return tuple(self.out_shapes.keys())

    @property
    @abstractmethod
    def in_shapes(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def out_shapes(self) -> Dict[str, Tuple[int, ...]]:
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
    def __init__(self, 
                 in_features: int, 
                 out_annotations: Annotations, 
                 exogenous: bool = False, 
                 *args, 
                 **kwargs):
        super().__init__(
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )
        self._in_features = in_features
        self.exogenous = exogenous
        in_shapes = self.in_shapes

    @property
    def in_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self.exogenous:
            return {"concept_embs": (self._in_features["concept_embs"],)}
        return {"residual": (self._in_features,)}

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
        if isinstance(x, ConceptTensor):
            # asssert the embedding field is not None and only one embedding is present
            # shape must be (batch_size, 1, emb_size)
            assert self.exogenous, f"Input to {self.__class__.__name__}.forward() cannot be a ConceptTensor unless exogenous=True."
            if x.concept_embs is None:
                raise ValueError(
                    f"The input ConceptTensor to {self.__class__.__name__}.forward() must have "
                    f"'concept_embs' not set to None."
                )
            # check shape
            if x.concept_embs.shape[1] != self.out_features['concept_probs'] or len(x.concept_embs.shape) != 3:
                raise ValueError(
                    f"The input ConceptTensor to {self.__class__.__name__}.forward() must have "
                    f"'concept_embs' of shape (batch_size, 1, {self.out_features['concept_probs']}), "
                    f"but got {x.concept_embs.shape} instead."
                )
            x = x.concept_embs  # shape (batch_size, n_concepts, emb_size)

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
    def __init__(self, in_features: Union[Tuple[Dict[str, int]], Dict[str, int]], out_annotations: Annotations, *args, **kwargs):
        super().__init__(
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )
        self._in_features = in_features
        in_shapes = self.in_shapes
        in_features = self.in_features
        out_shapes = self.out_shapes
        out_features = self.out_features

    @property
    def out_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"concept_probs": (self.out_probs_dim,)}

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
