import numpy as np
import torch

from torch_concepts import Annotations, ConceptTensor
from torch_concepts.nn.base.layer import BaseEncoder
from typing import List, Callable, Union, Dict, Tuple


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
        out_annotations: Annotations,
        embedding_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_annotations=out_annotations,
        )
        self.embedding_size = embedding_size
        # self.n_states = 2 # TODO: fix

        self.out_logits_dim = out_annotations.shape[1]
        self.out_exogenous_shape = (self.out_logits_dim, embedding_size) # * self.n_states)
        self.out_encoder_dim = np.prod(self.out_exogenous_shape).item()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                self.out_encoder_dim,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_exogenous_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self,
        embedding: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        return self.encoder(embedding)
