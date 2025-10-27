import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BaseEncoder
from typing import List, Callable, Union, Dict, Tuple


class ExogEncoder(BaseEncoder):
    """
    From latent code, creates one embedding per concept.
    Main reference: `"Concept Layer
    Models" <https://arxiv.org/pdf/2007.04612>`_

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
        activation: Callable = torch.nn.functional.leaky_relu,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_annotations=out_annotations,
        )

        self.activation = activation
        self.embedding_size = embedding_size

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features["residual"],
                self.embedding_size*self.out_probs_dim, # FIXME: fix for nonbinary concepts
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (self.out_probs_dim, self.embedding_size)),
        )

    @property
    def out_shapes(self) -> Dict[str, tuple]:
        return {"concept_embs": (self.embedding_size,)}

    def encode(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Encode concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ConceptTensor: Encoded concept scores.
        """
        c_logits = self.linear(x)
        c_embs = self.activation(c_logits)
        return ConceptTensor(self.out_annotations, concept_embs=c_embs)
