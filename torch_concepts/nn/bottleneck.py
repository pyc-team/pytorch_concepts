import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Union

from torch_concepts.base import ConceptTensor
from torch_concepts.nn import ConceptEncoder
from torch_concepts.nn.functional import intervene, concept_embedding_mixture


class BaseBottleneck(ABC, torch.Tensornn.Module):
    """
    BaseBottleneck is an abstract base class for concept bottlenecks.

    The concept dimension dictionary is structured as follows: keys are dimension
    indices and values are either integers (indicating the size of the
    dimension) or lists of strings (concept names).

    The output size is computed as the product of the sizes of all dimensions.
    The only exception is the batch size dimension, which is expected to be empty.

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
    def __init__(self, out_concept_dimensions: Dict[int, List[str]]):
        super().__init__()
        self.out_concept_dimensions = out_concept_dimensions

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, ConceptTensor]:
        pass


class ConceptBottleneck(BaseBottleneck):
    """
    ConceptBottleneck creates a bottleneck of supervised concept embeddings.
    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/pdf/2007.04612>`_

    The concept dimension dictionary should be structured as {1: n_concepts} or
    {1: ["concept_1_0", "concept_1_1", ..., "concept_1_n_concepts"]}.

    Attributes:
        in_features (int): Number of input features.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
        activation: Callable = F.sigmoid,
    ):
        super().__init__(out_concept_dimensions)
        self.scorer = ConceptEncoder(in_features, out_concept_dimensions)
        self.concept_names = self.scorer.concept_names
        self.output_size = self.scorer.output_size
        self.activation = activation

    def forward(
        self,
        x: ConceptTensor,
        c_true: ConceptTensor = None,
        intervention_idxs: ConceptTensor = None,
        intervention_rate: float = 0.0,
    ) -> Dict[str, ConceptTensor]:
        """
        Forward pass of ConceptBottleneck.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (ConceptTensor): Ground truth concepts.
            intervention_idxs (ConceptTensor): Boolean ConceptTensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            Dict[ConceptTensor]: 'next': object to pass to the next layer,
                'c_pred': concept scores with shape (batch_size, n_concepts),
                'c_int': concept scores after interventions, 'emb': None.
        """
        c_logit = self.scorer(x)
        c_pred = ConceptTensor.concept(
            self.activation(c_logit),
            self.concept_names,
        )
        c_int = intervene(c_pred, c_true, intervention_idxs)
        return dict(
            next=c_int,
            c_pred=c_pred,
            c_int=c_int,
            emb=None,
        )


class ConceptResidualBottleneck(BaseBottleneck):
    """
    ConceptResidualBottleneck is a layer where a first set of neurons is aligned
    with supervised concepts and a second set of neurons is free to encode
    residual information.
    Main reference: `"Promises and Pitfalls of Black-Box Concept Learning
    Models" <https://arxiv.org/abs/2106.13314>`_

    The concept dimension dictionary should be structured as {1: n_concepts} or
    {1: ["concept_1_0", "concept_1_1", ..., "concept_1_n_concepts"]}.

    Attributes:
        in_features (int): Number of input features.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept
            dimensions.
        residual_size (int): Size of residual embedding.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
        residual_size: int,
        activation: Callable = F.sigmoid,
    ):
        super().__init__(out_concept_dimensions)
        self.scorer = ConceptEncoder(in_features, out_concept_dimensions)
        self.concept_names = self.scorer.concept_names
        self.output_size = self.scorer.output_size + residual_size
        self.residual_size = residual_size
        self.residual_embedder = torch.Tensornn.Linear(
            in_features,
            residual_size,
        )
        self.activation = activation

    def forward(
        self,
        x: ConceptTensor,
        c_true: ConceptTensor = None,
        intervention_idxs: ConceptTensor = None,
        intervention_rate: float = 0.0,
    ) -> Dict[str, ConceptTensor]:
        """
        Forward pass of ConceptResidualBottleneck.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (ConceptTensor): Ground truth concepts.
            intervention_idxs (ConceptTensor): Boolean ConceptTensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            Dict[ConceptTensor]: 'next': object to pass to the next layer,
                'c_pred': concept scores with shape (batch_size, n_concepts),
                'c_int': concept scores after interventions, 'emb': residual
                embedding.
        """
        emb = self.residual_embedder(x)
        c_logit = self.scorer(x)
        c_pred = ConceptTensor.concept(self.activation(c_logit), self.concept_names)
        c_int = intervene(c_pred, c_true, intervention_idxs)
        return dict(
            next=torch.hstack((c_pred, emb)),
            c_pred=c_pred,
            c_int=c_int,
            emb=emb,
        )


class MixConceptEmbeddingBottleneck(BaseBottleneck):
    """
    MixConceptEmbeddingBottleneck creates supervised concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    The concept dimension dictionary should be structured as
    {1: n_concepts, 2: c_emb_size} or
    {1: ["concept_1_0", "concept_1_1", ..., "concept_1_n_concepts"],
    2: ["concept_2_0", "concept_2_1", ..., "concept_2_c_emb_size"]}.

    Attributes:
        in_features (int): Number of input features.
        out_concept_dimensions (Dict[int, Union[int, List[str]]]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features: int,
        out_concept_dimensions: Dict[int, Union[int, List[str]]],
        activation: Callable = F.sigmoid,
    ):
        super().__init__(out_concept_dimensions)
        self.encoder = ConceptEncoder(
            in_features=in_features,
            out_concept_dimensions=out_concept_dimensions,
        )
        self.scorer_in_size = (
            out_concept_dimensions[2]
            if isinstance(out_concept_dimensions[2], int)
            else len(out_concept_dimensions[2])
        )
        self.scorer = ConceptEncoder(
            in_features=self.scorer_in_size,
            out_concept_dimensions={1: out_concept_dimensions[1]},
            reduce_dim=2,
        )
        self.activation = activation
        self.concept_names = self.scorer.concept_names

    def forward(
        self,
        x: ConceptTensor,
        c_true: ConceptTensor = None,
        intervention_idxs: ConceptTensor = None,
        intervention_rate: float = 0.0,
    ) -> Dict[str, ConceptTensor]:
        """
        Forward pass of MixConceptEmbeddingBottleneck.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (ConceptTensor): Ground truth concepts.
            intervention_idxs (ConceptTensor): Boolean ConceptTensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            Dict[ConceptTensor]: 'next': object to pass to the next layer,
                'c_pred': concept scores with shape (batch_size, n_concepts),
                'c_int': concept scores after interventions, 'emb': residual
                embedding.
        """
        c_emb = self.encoder(x)
        c_logit = self.scorer(c_emb)
        c_pred = ConceptTensor.concept(
            self.activation(c_logit),
            self.concept_names,
        )
        c_int = intervene(c_pred, c_true, intervention_idxs)
        c_mix = concept_embedding_mixture(c_emb, c_int)
        return dict(next=c_mix,
            c_pred=c_pred,
            c_int=c_int,
            emb=None,
            context=c_emb,
        )

