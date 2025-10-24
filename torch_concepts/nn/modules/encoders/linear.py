import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BaseEncoderLayer
from typing import List, Callable, Union, Dict


class ProbEncoderLayer(BaseEncoderLayer):
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
        in_features: int,
        out_annotations: Annotations,
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_annotations=out_annotations,
        )

        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features,
                self.out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_shape),
        )

    @property
    def out_features(self) -> int:
        return self._out_concepts_size

    @property
    def out_shape(self) -> Union[torch.Size, tuple]:
        return self._out_concepts_shape

    @property
    def out_contract(self) -> Dict[str, int]:
        return {"concept_probs": self.out_features}

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
        c_probs = self.activation(c_logits)
        return ConceptTensor(self.out_annotations, concept_probs=c_probs)
