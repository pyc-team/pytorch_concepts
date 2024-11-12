import torch

from abc import ABC, abstractmethod
from typing import List, Dict, Union

from torch_concepts.base import ConceptTensor, ConceptDistribution
from torch_concepts.utils import validate_and_generate_concept_names, compute_output_size

EPS = 1e-8


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    The output objects are concept distributions or concept
    tensors with shape (batch_size, concept_dim1, ..., concept_dimN).

    The concept dimension dictionary is structured as follows: keys are
    dimension indices and values are either integers (indicating the size of the
    dimension) or lists of strings (concept names). The output size is computed
    as the product of the sizes of all dimensions. The only exception is the
    batch size dimension, which is expected to be empty.
    Example:
        {
            1: ["concept_a", "concept_b"],
            2: 3,
        }
    For 2 concepts in the first dimension and 3 concepts in the second
    dimension. This produces concept names:
        {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"],
        }
    The output size is computed as the product of the sizes of all dimensions
    except the batch size dimension. In this example, the output size is 6. The
    result of the forward is reshaped to match the concept names. Moreover, in
    this example the output shape is (batch_size, 2, 3).

    Attributes:
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
    """
    def __init__(
        self,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
    ):
        super().__init__()
        self.concept_names = validate_and_generate_concept_names(
            out_concept_dimensions
        )
        self.output_size = compute_output_size(self.concept_names)

    @abstractmethod
    def forward(self, x: ConceptTensor) -> Union[ConceptDistribution, ConceptTensor]:
        pass


class ConceptEncoder(BaseConceptLayer):
    """
    ConceptEncoder generates concept embeddings with shape
    (batch_size, concept_dim1, ..., concept_dimN).

    The concept dimension dictionary is structured as follows: keys are
    dimension indices and values are either integers (indicating the size of the
    dimension) or lists of strings (concept names). The output size is computed
    as the product of the sizes of all dimensions. The only exception is the
    batch size dimension, which is expected to be empty.
    Example:
        {
            1: ["concept_a", "concept_b"],
            2: 3,
        }
    For 2 concepts in the first dimension and 3 concepts in the second
    dimension. This produces concept names:
        {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"],
        }
    The output size is computed as the product of the sizes of all dimensions
    except the batch size dimension. In this example, the output size is 6. The
    result of the forward is reshaped to match the concept names. Moreover, in
    this example the output shape is (batch_size, 2, 3).

    Attributes:
        in_features (int): Number of input features.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
        reduce_dim (int): Dimension to eliminate from the output.
    """
    def __init__(
        self,
        in_features: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
        reduce_dim: int = None,
    ):
        super().__init__(out_concept_dimensions)
        self.in_features = in_features
        self.reduce_dim = reduce_dim
        if reduce_dim is not None:
            self.output_size = 1
        self.encoder = torch.nn.Linear(in_features, self.output_size)

    def forward(self, x: torch.Tensor) -> ConceptTensor:
        """
        Forward pass of the concept encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Concept embeddings with shape
                (batch_size, concept_dim1, ..., concept_dimN).
        """
        emb = self.encoder(x)
        concept_shape = tuple(
            len(self.concept_names[dim])
            for dim in sorted(self.concept_names.keys())
            if dim != 0
        )
        emb = emb.view(-1, *concept_shape)
        return ConceptTensor.concept(emb, self.concept_names.copy())


class ProbabilisticConceptEncoder(BaseConceptLayer):
    """
    ProbabilisticConceptEncoder generates concept context sampling from
    independent normal distributions. Samples are concept embeddings with shape
    (batch_size, concept_dim1, ..., concept_dimN).

    The concept dimension dictionary is structured as follows: keys are
    dimension indices and values are either integers (indicating the size of the
    dimension) or lists of strings (concept names). The output size is computed
    as the product of the sizes of all dimensions. The only exception is the
    batch size dimension, which is expected to be empty.
    Example:
        {
            1: ["concept_a", "concept_b"],
            2: 3,
        }
    For 2 concepts in the first dimension and 3 concepts in the second
    dimension. This produces concept names:
        {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"],
        }
    The output size is computed as the product of the sizes of all dimensions
    except the batch size dimension. In this example, the output size is 6. The
    result of the forward is reshaped to match the concept names. Moreover, in
    this example the output shape is (batch_size, 2, 3).


    Attributes:
        in_features (int): Number of input features.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
    """
    def __init__(
        self,
        in_features: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
    ):
        super().__init__(out_concept_dimensions)
        self.in_features = in_features
        self.concept_mean_predictor = torch.nn.Linear(
            in_features,
            self.output_size,
        )
        self.concept_var_predictor = torch.nn.Linear(
            in_features,
            self.output_size,
        )

    def forward(self, x: torch.Tensor) -> ConceptDistribution:
        """
        Forward pass of the concept encoder with sampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Concept distribution whose samples are concept
                embeddings with shape (batch_size, concept_dim1, ...,
                concept_dimN).
        """
        z_mu = self.concept_mean_predictor(x)
        z_log_var = self.concept_var_predictor(x)

        # reshape to match concept names
        concept_shape = tuple(
            len(self.concept_names[dim])
            for dim in sorted(self.concept_names.keys())
            if dim != 0
        )
        z_mu = z_mu.view(-1, *concept_shape)
        z_log_var = z_log_var.view(-1, *concept_shape)

        z_sigma = torch.exp(z_log_var / 2) + EPS
        qz_x = torch.distributions.Normal(z_mu, z_sigma)
        p_z = torch.distributions.Normal(
            torch.zeros_like(qz_x.mean),
            torch.ones_like(qz_x.variance)
        )
        self.p_z = ConceptDistribution(p_z, self.concept_names.copy())
        return ConceptDistribution(qz_x, self.concept_names.copy())


class ConceptMemory(torch.nn.Module):
    """
    ConceptMemory is a memory module that contains a set of embeddings which can
    be decoded into different concept states. The output objects are concept
    tensors with shape (memory_size, concept_dim1, ..., concept_dimN).

    The concept dimension dictionary is structured as follows: keys are
    dimension indices and values are either integers (indicating the size of the
    dimension) or lists of strings (concept names). The output size is computed
    as the product of the sizes of all dimensions. The only exception is the
    batch size dimension, which is expected to be empty.
    Example:
        {
            1: ["concept_a", "concept_b"],
            2: 3,
        }
    For 2 concepts in the first dimension and 3 concepts in the second
    dimension. This produces concept names:
        {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"],
        }
    The output size is computed as the product of the sizes of all dimensions
    except the batch size dimension. In this example, the output size is 6. The
    result of the forward is reshaped to match the concept names. Moreover, in
    this example the output shape is (batch_size, 2, 3).

    Attributes:
        memory_size (int): Number of elements in the memory.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
    """
    def __init__(
        self,
        memory_size: int,
        emb_size: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
    ):
        super().__init__()
        self.concept_names = validate_and_generate_concept_names(
            out_concept_dimensions
        )
        self.output_size = compute_output_size(self.concept_names)
        self.memory_size = memory_size
        self.emb_size = emb_size
        self.latent_memory = torch.nn.Embedding(self.memory_size, self.emb_size)
        self.memory_decoder = torch.nn.Linear(self.emb_size, self.output_size)

    def forward(self, idxs: torch.Tensor = None) -> ConceptTensor:
        """
        Forward pass of the concept memory.

        Args:
            idxs (Tensor): Indices of rules to evaluate with shape
                (batch_size, n_tasks). Default is None (evaluate all).

        Returns:
            ConceptTensor: Concept roles with shape (memory_size, n_concepts,
                n_tasks, n_concept_states).
        """
        memory_embs = self.latent_memory.weight
        concept_weights = self.memory_decoder(memory_embs)
        concept_shape = tuple(
            len(self.concept_names[dim])
            for dim in sorted(self.concept_names.keys())
            if dim != 0
        )
        concept_weights = concept_weights.view(-1, *concept_shape)
        return ConceptTensor.concept(concept_weights, self.concept_names.copy())
