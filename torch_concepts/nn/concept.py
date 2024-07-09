import torch
from torch import nn
from abc import ABC, abstractmethod

from torch_concepts.base import ConceptTensor


class ConceptEncoder(nn.Module):
    """
    ConceptEncoder generates concept embeddings.

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        emb_size (int): Size of concept embeddings.
    """
    def __init__(self, in_features, n_concepts, emb_size=1):
        super().__init__()
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.in_features = in_features
        self.encoder = nn.Linear(in_features, emb_size * n_concepts)

    def forward(self, x: torch.Tensor) -> ConceptTensor:
        """
        Forward pass of the concept encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Concept embeddings. If emb_size is 1, the returned shape is (batch_size, n_concepts). Otherwise, the shape is (batch_size, n_concepts, emb_size).
        """
        emb = ConceptTensor.concept(self.encoder(x))
        if self.emb_size > 1:
            return emb.view(-1, self.n_concepts, self.emb_size)
        return emb


class BaseConceptLayer(ABC, nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: ConceptTensor) -> ConceptTensor:
        pass


class ConceptScorer(BaseConceptLayer):
    """
    ConceptScorer scores concept embeddings.

    Attributes:
        emb_size (int): Size of concept embeddings.
    """
    def __init__(self, emb_size):
        super().__init__()
        self.scorer = nn.Linear(emb_size, 1)

    def forward(self, x: ConceptTensor) -> ConceptTensor:
        """
        Forward pass of the concept scorer.

        Args:
            x (ConceptTensor): Concept embeddings with shape (batch_size, n_concepts, emb_size).

        Returns:
            ConceptTensor: Concept scores with shape (batch_size, n_concepts).
        """
        return self.scorer(x).squeeze(-1)
