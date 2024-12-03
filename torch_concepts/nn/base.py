import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import List, Union

from torch_concepts.base import ConceptTensor, ConceptDistribution


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    The output objects are concept distributions or concept
    tensors with shape (batch_size, n_concepts, concept_dim1, ..., concept_dimN).
    """
    def __init__(
        self,
        concepts: Union[List[str], int],
        emb_shape: List[int] = None,
    ):
        super().__init__()
        if isinstance(concepts, list):
            concept_names = concepts
            n_concepts = len(concepts)
        else:
            n_concepts = concepts
            concept_names = concept_names = [
                f'concept_{i}' for i in range(n_concepts)
            ]
        self.emb_shape = emb_shape or []
        shape = [n_concepts] + self.emb_shape
        self.n_concepts = n_concepts
        self.concept_name = concept_names
        self.concept_axis = 0
        self._shape = shape
        self.output_size = np.prod(self.shape())

    def shape(self):
        return self._shape

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError('transform')

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[ConceptDistribution, ConceptTensor]:
        return ConceptTensor.tensor(
            tensor=self.transform(x),
            concept_names=self.concept_names,
            concept_axis=self.concept_axis,
        )


class LinearConceptLayer(BaseConceptLayer):
    """
    LinearConceptLayer generates concept embeddings with shape
    (batch_size, n_concepts, concept_dim1, ..., concept_dimN).

    """
    def __init__(
        self,
        in_features: int,
        concepts: Union[List[str], int],
        emb_shape: List[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            concepts=concepts,
            emb_shape=emb_shape,
        )
        self.in_features = in_features
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )


class ProbabilisticConceptLayer(BaseConceptLayer):
    """
    ProbabilisticConceptLayer generates concept context sampling from
    independent normal distributions. Samples are concept embeddings with shape
    (batch_size, n_concepts, concept_dim1, ..., concept_dimN).

    """
    def __init__(
        self,
        in_features: int,
        concepts: Union[List[str], int],
        emb_shape: List[int] = None,
        eps=1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(
            concepts=concepts,
            emb_shape=emb_shape,
        )
        self.in_features = in_features
        self.eps = eps
        self.concept_mean_predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )
        self.concept_var_predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )

    def transform(self, x: torch.Tensor) -> ConceptDistribution:
        z_mu = self.concept_mean_predictor(x)
        z_log_var = self.concept_var_predictor(x)

        z_sigma = torch.exp(z_log_var / 2) + self.eps
        qz_x = torch.distributions.Normal(z_mu, z_sigma)
        return ConceptDistribution(
            base_dist=qz_x,
            concept_names=self.concept_names,
            concept_axis=self.concept_axis,
        )


class ConceptMemory(torch.nn.Module):
    """
    ConceptMemory is a memory module that contains a set of embeddings which can
    be decoded into different concept states. The output objects are concept
    tensors with shape (memory_size, concept_dim1, ..., concept_dimN).

    """
    def __init__(
        self,
        memory_size: int,
        out_features: Union[List[int], int],
        concepts: Union[List[str], int],
        emb_shape: List[int] = None,
        eps=1e-8,
        *args,
        **kwargs,
    ):
        super().__init__(
            concepts=concepts,
            emb_shape=emb_shape,
        )
        if isinstance(out_features, int):
            out_features = [out_features]
        self.out_features = out_features
        self.output_size = np.prod(out_features)
        self.memory_size = memory_size
        if len(self.emb_shape) == 0:
            raise ValueError(
                f'ConceptMemory layer needs a non-zero embedding size for '
                f'each concept.'
            )
        self.latent_memory = torch.nn.Embedding(
            self.memory_size,
            np.prod(self.emb_shape),
        )
        self.memory_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                np.prod(self.emb_shape),
                np.prod(self.out_features),
            ),
            torch.nn.Unflatten(-1, self.out_features),
        )

    def transform(self, x: torch.Tensor = None) -> ConceptTensor:
        return ConceptTensor.concept(
            tensor=self.memory_decoder(self.latent_memory.weight),
            concept_names=self.concept_names,
            concept_idx=self.concept_idx,
        )