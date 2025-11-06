from typing import Callable

import torch

from abc import ABC


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    """

    def __init__(
        self,
        out_features: int,
        in_features_logits: int = None,
        in_features_embedding: int = None,
        in_features_exogenous: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features_logits = in_features_logits
        self.in_features_embedding = in_features_embedding
        self.in_features_exogenous = in_features_exogenous
        self.out_features = out_features

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class BaseEncoder(BaseConceptLayer):
    """
    BaseConceptLayer is an abstract base class for concept encoder layers.
    The output objects are ConceptTensors.
    """
    def __init__(self,
                 out_features: int,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None):
        super().__init__(
            in_features_logits=None,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features
        )


class BasePredictor(BaseConceptLayer):
    """
    BasePredictor is an abstract base class for concept predictor layers.
    The input objects are ConceptTensors and the output objects are ConceptTensors with concept probabilities only.
    """
    def __init__(self,
                 out_features: int,
                 in_features_logits: int,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None,
                 in_activation: Callable = torch.sigmoid):
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features,
        )
        self.in_activation = in_activation
