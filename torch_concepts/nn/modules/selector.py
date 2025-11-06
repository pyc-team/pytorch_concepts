import numpy as np
import torch
import torch.nn.functional as F


from ..base.layer import BaseEncoder
from typing import List, Union


class MemorySelector(BaseEncoder):
    """
    ConceptLayer creates a bottleneck of supervised concepts.
    Main reference: `"Concept Layer
    Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """
    def __init__(
        self,
        in_features_embedding: int,
        memory_size : int,
        embedding_size: int,
        out_features: int,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.temperature = temperature
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self._annotation_out_features = out_features
        self._embedding_out_features = memory_size * embedding_size
        self._selector_out_shape = (self._annotation_out_features, memory_size)
        self._selector_out_features = np.prod(self._selector_out_shape).item()

        # init memory of embeddings [out_features, memory_size * embedding_size]
        self.memory = torch.nn.Embedding(self._annotation_out_features, self._embedding_out_features)

        # init selector [B, out_features]
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(in_features_embedding, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                embedding_size,
                self._selector_out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        )

    def forward(
        self,
        embedding: torch.Tensor = None,
        sampling: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        memory = self.memory.weight.view(-1, self.memory_size, self.embedding_size)
        logits = self.selector(embedding)
        if sampling:
            probs = F.gumbel_softmax(logits, dim=1, tau=self.temperature, hard=True)
        else:
            probs = torch.softmax(logits / self.temperature, dim=1)

        exogenous = torch.einsum("btm,tme->bte", probs, memory) # [Batch x Task x Memory] x [Task x Memory x Emb] -> [Batch x Task x Emb]
        return exogenous
