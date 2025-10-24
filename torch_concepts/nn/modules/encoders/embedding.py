import numpy as np
import torch

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor
from ...base.layer import BaseEncoderLayer
from typing import List, Dict, Callable, Union, Tuple


class ProbEmbEncoderLayer(BaseEncoderLayer):
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
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_annotations=out_annotations,
        )
        self.activation = activation
        self.embedding_size = embedding_size

        self._out_concepts_shape = (self.out_annotations.shape[1], embedding_size * 2)
        self._out_concepts_size = np.prod(self._out_concepts_shape).item()

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features,
                self.out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_shape),
            torch.nn.LeakyReLU(),
        )
        self.concept_score_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(self.out_shape[-1], 1),
            torch.nn.Flatten(),
        )

    @property
    def out_features(self) -> int:
        return self._out_concepts_size

    @property
    def out_shape(self) -> Union[torch.Size, tuple]:
        return self._out_concepts_shape

    @property
    def out_contract(self) -> Dict[str, int]:
        return {"concept_probs": self.out_shape[0], "concept_embs": self.out_shape}

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
