from typing import List

import torch
from torch import nn
from abc import ABC, abstractmethod

from torch_concepts.base import ConceptTensor, ConceptDistribution

EPS = 1e-8


class ConceptEncoder(nn.Module):
    """
    ConceptEncoder generates concept embeddings.

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        emb_size (int): Size of concept embeddings.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, in_features: int, n_concepts: int, emb_size: int = 1, concept_names: List[str] = None):
        super().__init__()
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.in_features = in_features
        self.encoder = nn.Linear(in_features, emb_size * n_concepts)
        self.concept_names = concept_names

    def forward(self, x: torch.Tensor) -> ConceptTensor:
        """
        Forward pass of the concept encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Concept embeddings. If emb_size is 1, the returned shape is (batch_size, n_concepts). Otherwise, the shape is (batch_size, n_concepts, emb_size).
        """
        emb = self.encoder(x)
        if self.emb_size > 1:
            emb = emb.view(-1, self.n_concepts, self.emb_size)
        return ConceptTensor.concept(emb, self.concept_names)


class ProbabilisticConceptEncoder(ConceptEncoder):
    """
    ProbabilisticConceptEncoder generates concept context sampling from independent normal distributions.

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        emb_size (int): Size of concept embeddings.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, in_features: int, n_concepts: int, emb_size: int = 1, concept_names: List[str] = None):
        super().__init__(in_features, n_concepts, emb_size, concept_names)
        self.concept_mean_predictor = nn.Linear(in_features, emb_size * n_concepts)
        self.concept_var_predictor = nn.Linear(in_features, emb_size * n_concepts)

    def forward(self, x: torch.Tensor) -> ConceptDistribution:
        """
        Forward pass of the concept encoder with sampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Concept embeddings. If emb_size is 1, the returned shape is (batch_size, n_concepts). Otherwise, the shape is (batch_size, n_concepts, emb_size).
        """
        z_mu = self.concept_mean_predictor(x)
        z_log_var = self.concept_var_predictor(x)
        if self.emb_size > 1:
            z_mu = z_mu.view(-1, self.n_concepts, self.emb_size)
            z_log_var = z_log_var.view(-1, self.n_concepts, self.emb_size)

        z_sigma = torch.exp(z_log_var / 2) + EPS
        qz_x = torch.distributions.Normal(z_mu, z_sigma)
        p_z = torch.distributions.Normal(torch.zeros_like(qz_x.mean), torch.ones_like(qz_x.variance))
        self.p_z = ConceptDistribution(p_z, self.concept_names)
        return ConceptDistribution(qz_x, self.concept_names)


class BaseConceptLayer(ABC, nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.

    Attributes:
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, concept_names: List[str] = None):
        super().__init__()
        self.concept_names = concept_names

    @abstractmethod
    def forward(self, x: ConceptTensor) -> ConceptTensor:
        pass


class ConceptScorer(BaseConceptLayer):
    """
    ConceptScorer scores concept embeddings.

    Attributes:
        emb_size (int): Size of concept embeddings.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, emb_size: int, concept_names: List[str] = None):
        super().__init__(concept_names)
        self.scorer = nn.Linear(emb_size, 1)

    def forward(self, x: ConceptTensor) -> ConceptTensor:
        """
        Forward pass of the concept scorer.

        Args:
            x (ConceptTensor): Concept embeddings with shape (batch_size, n_concepts, emb_size).

        Returns:
            ConceptTensor: Concept scores with shape (batch_size, n_concepts).
        """
        return ConceptTensor.concept(self.scorer(x).squeeze(-1), self.concept_names)


class ConceptMemory(nn.Module):
    """
    ConceptMemory is a memory module that contains a set of embeddings which can be decoded into different concept states.

    Attributes:
        n_concepts (int): Number of concepts.
        n_tasks (int): Number of tasks.
        memory_size (int): Number of elements in the memory.
        emb_size (int): Size of each element of the memory.
        n_concept_states (int): Number of states for each concept.
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, n_concepts: int, n_tasks: int, memory_size: int, emb_size: int, n_concept_states: int = 3, concept_names: List[str] = None):
        super().__init__()
        self.concept_names = concept_names
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.memory_size = memory_size
        self.emb_size = emb_size
        self.n_concept_states = n_concept_states
        self.latent_memory = torch.nn.Embedding(self.memory_size * self.n_tasks,  self.emb_size)
        self.memory_decoder = torch.nn.Linear(self.emb_size, self.n_concepts * self.n_concept_states)

    def forward(self, idxs: torch.Tensor = None) -> ConceptTensor:
        """
        Forward pass of the concept memory.

        Args:
            idxs (Tensor): Indices of rules to evaluate with shape (batch_size, n_tasks). Default is None (evaluate all).

        Returns:
            ConceptTensor: Concept roles with shape (memory_size, n_concepts, n_tasks, n_concept_states).
        """
        memory_embs = self.latent_memory.weight
        if idxs is None:
            concept_weights = self.memory_decoder(memory_embs).view(self.memory_size, self.n_concepts, self.n_tasks, self.n_concept_states)

        else:
            # TODO: to check if this is still correct
            idxs = idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
            rule_embs = memory_embs.unsqueeze(0).expand(idxs.size(0), -1, -1, -1)  # add batch dimension
            rule_embs = torch.gather(rule_embs, 2, idxs).squeeze(2)
            concept_weights = self.memory_decoder(rule_embs).view(-1, self.n_concepts, self.n_tasks, self.n_concept_states).unsqueeze(2)

        return ConceptTensor.concept(concept_weights, self.concept_names)
