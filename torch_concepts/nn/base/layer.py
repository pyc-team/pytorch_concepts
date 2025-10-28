from typing import Union, Dict, Tuple, Callable

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
        in_features_logits: int = None,
        in_features_embedding: int = None,
        in_features_exogenous: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.out_annotations = out_annotations
        self.in_features_logits = in_features_logits
        self.in_features_embedding = in_features_embedding
        self.in_features_exogenous = in_features_exogenous

        self.concept_axis = 1
        self.out_probs_dim = out_annotations.shape[1]

    def forward(
        self,
        logits: torch.Tensor = None,
        embedding: torch.Tensor = None,
        exogenous: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
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
                 out_annotations: Annotations,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None,
                 *args, 
                 **kwargs):
        super().__init__(
            in_features_logits=None,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )


class BasePredictor(BaseConceptLayer):
    """
    BasePredictor is an abstract base class for concept predictor layers.
    The input objects are ConceptTensors and the output objects are ConceptTensors with concept probabilities only.
    """
    def __init__(self,
                 out_annotations: Annotations,
                 in_features_logits: int,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None,
                 in_activation: Callable = torch.sigmoid,
                 *args, 
                 **kwargs):
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_annotations=out_annotations,
            *args,
            **kwargs,
        )
        self.in_activation = in_activation
