from collections import defaultdict
from typing import List, Tuple

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


class Memory(ABC, nn.Module):
    """
    Memory is an abstract class for memories that contain modules to be evaluated with concepts.

    Attributes:
        n_concepts (int): Number of concepts.
        n_tasks (int): Number of tasks.
        memory_size (int): Size of memory (in number of modules).
        concept_names (List[str]): Names of concepts.
    """
    def __init__(self, n_concepts: int, n_tasks: int, memory_size: int, concept_names: List[str] = None):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.memory_size = memory_size
        self.concept_names = concept_names

    @abstractmethod
    def forward(self, x: ConceptTensor) -> ConceptTensor:
        pass


class LogicMemory(Memory):
    """
    LogicMemory is a memory module that contains rules for reasoning with concepts.

    Attributes:
        n_concepts (int): Number of concepts.
        n_tasks (int): Number of tasks.
        memory_size (int): Size of memory (in number of rules).
        rule_emb_size (int): Size of rule embeddings.
        concept_names (List[str]): Names of concepts
    """
    def __init__(self, n_concepts: int, n_tasks: int, memory_size: int, rule_emb_size: int, concept_names: List[str] = None):
        super().__init__(n_concepts, n_tasks, memory_size, concept_names=concept_names)
        self.rule_emb_size = rule_emb_size
        self.rule_embeddings = torch.nn.Embedding(self.memory_size, self.n_tasks * self.rule_emb_size)
        self.rule_decoder = torch.nn.Linear(self.rule_emb_size, self.n_concepts * 3)

    def _decode_rules(self, idxs=None):
        """
        Decodes rules from rule embeddings.

        Args:
            idxs (Tensor): Indices of rules to decode with shape (batch_size, n_tasks). Default is None (decode all).

        Returns:
            torch.Tensor: Rule logits with shape (n_tasks, memory_size, n_concepts, 3).
        """
        rule_embs = self.rule_embeddings.weight.view(self.n_tasks, self.memory_size, self.rule_emb_size)
        if idxs is None:
            logits = self.rule_decoder(rule_embs).view(self.n_tasks, self.memory_size, self.n_concepts, 3)
        else:
            idxs = idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.rule_emb_size)
            rule_embs = rule_embs.unsqueeze(0).expand(idxs.size(0), -1, -1, -1)  # add batch dimension
            rule_embs = torch.gather(rule_embs, 2, idxs).squeeze(2)
            logits = self.rule_decoder(rule_embs).view(-1, self.n_tasks, self.n_concepts, 3).unsqueeze(2)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: ConceptTensor, idxs=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the logic memory.

        Args:
            x (ConceptTensor): Concept predictions with shape (batch_size, n_concepts).
            idxs (Tensor): Indices of rules to evaluate with shape (batch_size, n_tasks). Default is None (evaluate all).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of rule predictions with shape (batch_size, n_tasks, memory_size) and concept reconstructions with shape (batch_size, n_tasks, memory_size, n_concepts).
        """
        concept_roles = self._decode_rules(idxs)  # (batch_size,) n_tasks, memory_size, n_concepts, 3
        pos_polarity, neg_polarity, irrelevance = concept_roles[..., 0], concept_roles[..., 1], concept_roles[..., 2]

        if idxs is None:  # cast all to (batch_size, n_tasks, memory_size, n_concepts)
            x = x.unsqueeze(1).unsqueeze(1).expand(-1, self.n_tasks, self.memory_size, -1)
            pos_polarity = pos_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
            neg_polarity = neg_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
            irrelevance = irrelevance.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        else:  # cast all to (batch_size, n_tasks, 1, n_concepts)
            x = x.unsqueeze(1).unsqueeze(1).expand(-1, self.n_tasks, 1, -1)

        # TODO: incorporate t-norms?
        y_per_rule = (irrelevance + (1-x) * neg_polarity + x * pos_polarity).prod(dim=-1)  # batch_size, n_tasks, mem_size

        c_reconstruction = 0.5 * irrelevance + pos_polarity  # batch_size, n_tasks, mem_size, n_concepts

        return y_per_rule, c_reconstruction

    def extract_rules(self):
        """
        Extracts rules from rule embeddings as strings.

        Returns:
            Dict[str, Dict[str, str]]: Rules as strings.
        """
        rules_str = defaultdict(dict)  # task, memory_size
        rules = self._decode_rules().detach().cpu()  # n_tasks, memory_size, n_concepts, 3
        roles = torch.argmax(rules, dim=-1)  # memory_size, n_tasks, n_concepts
        for task_idx in range(self.n_tasks):
            for mem_idx in range(self.memory_size):
                rule = [("~ " if roles[task_idx, mem_idx, concept_idx] == 1 else "") + self.concept_names[concept_idx]
                        for concept_idx in range(self.n_concepts)
                            if roles[task_idx, mem_idx, concept_idx] != 2]
                rules_str[f"Task {task_idx}"][f"Rule {mem_idx}"] = " & ".join(rule)
        return rules_str
