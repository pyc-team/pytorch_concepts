import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BaseEncoder
from typing import List, Callable, Union, Dict, Tuple


class ProbEncoder(BaseEncoder):
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
                self.in_concept_features["residual"],
                self.out_concept_features["concept_probs"],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_concept_shapes["concept_probs"]),
        )

    @property
    def out_concept_shapes(self) -> Dict[str, tuple]:
        return {"concept_probs": (self.out_probs_dim,)}

    @property
    def out_concepts(self) -> Tuple[str]:
        return ("concept_probs",)

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
