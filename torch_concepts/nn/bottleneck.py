import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Callable, Union, Tuple

from torch_concepts.base import AnnotatedTensor
from torch_concepts.nn import Annotate
from torch_concepts.nn.functional import intervene, concept_embedding_mixture


def _check_annotations(annotations: Union[List[str], int]):
    assert isinstance(annotations, (list, int)), \
        "annotations must be either a single list of str or a single int"
    if isinstance(annotations, list):
        assert all(isinstance(a, str) for a in annotations), \
            "all elements in the annotations list must be of type str"


class BaseConceptBottleneck(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    The output objects are annotated tensors.
    """
    def __init__(
        self,
        in_features: int,
        annotations: List[Union[List[str], int]],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features

        self.annotations = []
        shape = []
        self.annotated_axes = []
        for dim, annotation in enumerate(annotations):
            if isinstance(annotation, int):
                shape.append(annotation)
            else:
                self.annotations.append(annotation)
                shape.append(len(annotation))
                self.annotated_axes.append(dim+1)

        self.concept_axis = 1
        self._shape = shape
        self.output_size = np.prod(self.shape())

        self.annotator = Annotate(self.annotations, self.annotated_axes)

    def shape(self):
        return self._shape

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        raise NotImplementedError('predict')

    @abstractmethod
    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        raise NotImplementedError('intervene')

    @abstractmethod
    def transform(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed tensor and dictionary with
                intermediate concepts tensors.
        """
        raise NotImplementedError('transform')

    def annotate(
        self,
        x: torch.Tensor,
    ) -> AnnotatedTensor:
        """
        Annotate tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            AnnotatedTensor: Annotated tensor.
        """
        return self.annotator(x)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Forward pass of a ConceptBottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor
                and dictionary with intermediate concepts tensors.
        """
        x_new, val_dict = self.transform(x, *args, **kwargs)
        return x_new, val_dict


class LinearConceptBottleneck(BaseConceptBottleneck):
    """
    ConceptBottleneck creates a bottleneck of supervised concepts.
    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        activation: Callable = F.sigmoid,
        *args,
        **kwargs,
    ):
        _check_annotations(annotations)

        if isinstance(annotations, int):
            annotations = [annotations]

        super().__init__(
            in_features=in_features,
            annotations=[annotations],
        )
        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_emb = self.linear(x)
        return self.activation(c_emb)

    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        return intervene(x, c_true, intervention_idxs)

    def transform(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_pred = c_int = self.predict(x)
        if 'c_true' in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        return c_int, dict(c_pred=c_pred, c_int=c_int)


class LinearConceptResidualBottleneck(LinearConceptBottleneck):
    """
    ConceptResidualBottleneck is a layer where a first set of neurons is aligned
    with supervised concepts and a second set of neurons is free to encode
    residual information.
    Main reference: `"Promises and Pitfalls of Black-Box Concept Learning
    Models" <https://arxiv.org/abs/2106.13314>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        residual_size: int,
        activation: Callable = F.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            annotations=annotations,
            activation=activation,
            *args,
            **kwargs,
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_features, residual_size),
            torch.nn.LeakyReLU()
        )
        self.annotations_extended = copy.deepcopy(self.annotations)
        self.annotations_extended[0].extend(
            [f"residual_{i}" for i in range(residual_size)]
        )
        self.annotator_extended = Annotate(
            self.annotations_extended,
            self.annotated_axes,
        )

    def transform(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_pred = c_int = self.predict(x)
        emb = self.residual(x)
        if 'c_true' in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        c_new = torch.hstack((c_pred, emb))
        c_new = self.annotator_extended(c_new)
        return c_new, dict(c_pred=c_pred, c_int=c_int)


class ConceptEmbeddingBottleneck(BaseConceptBottleneck):
    """
    ConceptEmbeddingBottleneck creates supervised concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        embedding_size: int,
        activation: Callable = F.sigmoid,
        *args,
        **kwargs,
    ):
        _check_annotations(annotations)
        annotations = [annotations, embedding_size]
        n_concepts = (
            len(annotations[0]) if isinstance(annotations[0], list)
            else annotations[0]
        )

        super().__init__(
            in_features=in_features,
            annotations=annotations,
        )

        self._shape = [n_concepts, embedding_size * 2]
        self.output_size = np.prod(self.shape())

        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
            torch.nn.LeakyReLU(),
        )
        self.concept_score_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(self.shape()[-1], 1),
            torch.nn.Flatten(),
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_emb = self.linear(x)
        return self.activation(self.concept_score_bottleneck(c_emb))

    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        return intervene(x, c_true, intervention_idxs)

    def transform(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_emb = self.linear(x)
        c_pred = c_int = self.activation(self.concept_score_bottleneck(c_emb))
        if 'c_true' in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_mix = concept_embedding_mixture(c_emb, c_int)
        c_mix = self.annotate(c_mix)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        return c_mix, dict(c_pred=c_pred, c_int=c_int)
