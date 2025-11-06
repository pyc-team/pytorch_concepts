import numpy as np
import torch

from torch_concepts.nn.base.layer import BaseEncoder
from typing import List, Union, Tuple


class ExogEncoder(BaseEncoder):
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
        in_features_embedding: int,
        out_features: int,
        embedding_size: int
    ):
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.embedding_size = embedding_size

        self.out_logits_dim = out_features
        self.out_exogenous_shape = (self.out_logits_dim, embedding_size)
        self.out_encoder_dim = np.prod(self.out_exogenous_shape).item()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                self.out_encoder_dim
            ),
            torch.nn.Unflatten(-1, self.out_exogenous_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self,
        embedding: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        return self.encoder(embedding)
