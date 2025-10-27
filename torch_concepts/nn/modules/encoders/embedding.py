import numpy as np
import torch

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor
from ...base.layer import BaseEncoder
from typing import List, Dict, Callable, Union, Tuple


class ProbEmbEncoder(BaseEncoder):
    """
    ConceptEmbeddingLayer creates supervised concept embeddings.
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
        out_annotations: Annotations,
        embedding_size: int,
        exogenous: bool = False,
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_annotations=out_annotations,
            exogenous=exogenous,
        )
        self.activation = activation
        self.embedding_size = embedding_size

        self.n_states = 2 # TODO: fix
        self.out_concept_emb_shapes = (self.out_probs_dim, embedding_size * self.n_states)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features["residual"],
                self.out_features["concept_embs"],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_shapes["concept_embs"]),
            torch.nn.LeakyReLU(),
        )
        self.concept_score_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(self.out_shapes["concept_embs"][1], 1), # FIXME: check for different types of concepts
            torch.nn.Flatten(),
        )

    @property
    def out_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"concept_embs": self.out_concept_emb_shapes, "concept_probs": (self.out_probs_dim,)}

    def encode(
        self, x: torch.Tensor, *args, **kwargs
    ) -> ConceptTensor:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_emb = self.linear(x)
        c_probs = self.activation(self.concept_score_bottleneck(c_emb))
        return ConceptTensor(self.out_annotations, concept_probs=c_probs, concept_embs=c_emb)
