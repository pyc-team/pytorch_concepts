from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import torch
from torch import nn
import torch.nn.functional as F

from torch_concepts.base import ConceptTensor
from torch_concepts.nn import ConceptEncoder, ConceptScorer
from torch_concepts.nn.functional import intervene


class BaseBottleneck(ABC, nn.Module):
    """
    BaseBottleneck is an abstract base class for concept bottlenecks.

    Attributes:
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, concept_names: List[str] = None):
        super().__init__()
        self.concept_names = concept_names

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, ConceptTensor]:
        pass


class ConceptBottleneck(BaseBottleneck):
    """
    ConceptBottleneck creates a bottleneck of supervised concept embeddings.
    Main reference: `"Concept Bottleneck Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, in_features: int, n_concepts: int, concept_names: List[str] = None,
                 activation: Callable = F.sigmoid):
        super().__init__(concept_names)
        self.scorer = ConceptEncoder(in_features, n_concepts, 1, concept_names)
        self.activation = activation

    def forward(self, x: ConceptTensor, c_true: ConceptTensor = None, intervention_idxs: ConceptTensor = None,
                intervention_rate: float = 0.0) -> Dict[str, ConceptTensor]:
        """
        Forward pass of ConceptBottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Dict[ConceptTensor]: 'next': object to pass to the next layer, 'c_pred': concept scores with shape (batch_size, n_concepts), 'c_int': concept scores after interventions, 'emb': None.
        """
        c_logit = self.scorer(x)
        c_pred = self.activation(c_logit)

        # intervene
        c_int = ConceptTensor.concept(c_pred.clone(), self.concept_names)
        if c_true is not None and intervention_idxs is not None:
            if intervention_idxs.min() >= 0 and intervention_idxs.max() >= self.n_concepts:
                raise ValueError("Intervention indices must be less than the number of concepts.")

            c_int = ConceptTensor.concept(intervene(c_pred, c_true, intervention_idxs))

        return {'next': c_int, 'c_pred': c_pred, 'c_int': c_int, 'emb': None}


class ConceptResidualBottleneck(BaseBottleneck):
    """
    ConceptResidualBottleneck is a layer where a first set of neurons is aligned with supervised concepts and
    a second set of neurons is free to encode residual information.
    Main reference: `"Promises and Pitfalls of Black-Box Concept Learning Models" <https://arxiv.org/abs/2106.13314>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        emb_size (int): Size of concept embeddings.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, in_features: int, n_concepts: int, emb_size: int, concept_names: List[str] = None,
                 activation: Callable = F.sigmoid):
        super().__init__(concept_names)
        self.scorer = ConceptEncoder(in_features, n_concepts, 1, concept_names)
        self.residual_embedder = nn.Linear(in_features, emb_size)
        self.activation = activation

    def forward(self, x: ConceptTensor, c_true: ConceptTensor = None, intervention_idxs: ConceptTensor = None,
                intervention_rate: float = 0.0) -> Dict[str, ConceptTensor]:
        """
        Forward pass of ConceptResidualBottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Dict[ConceptTensor]: 'next': object to pass to the next layer, 'c_pred': concept scores with shape (batch_size, n_concepts), 'c_int': concept scores after interventions, 'emb': residual embedding.
        """
        emb = self.residual_embedder(x)
        c_logit = self.scorer(x)
        c_pred = ConceptTensor.concept(self.activation(c_logit), self.concept_names)

        # intervene
        c_int = c_pred.clone()
        if c_true is not None and intervention_idxs is not None:
            if intervention_idxs.max() >= self.n_concepts:
                raise ValueError("Intervention indices must be less than the number of concepts.")

            c_int = intervene(c_pred, c_true, intervention_idxs)

        return {'next': torch.hstack((c_pred, emb)), 'c_pred': c_pred, 'c_int': c_int, 'emb': emb}

