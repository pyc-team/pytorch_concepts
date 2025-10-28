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
        in_features_global: int,
        in_features_exogenous: int,
        out_annotations: Annotations,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_global=in_features_global,
            in_features_exogenous=in_features_exogenous,
            out_annotations=out_annotations,
        )

        self.exogenous_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_exogenous,
                1,
                *args,
                **kwargs,
            ),
            torch.nn.Flatten(),
        )
        self.global_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_global,
                self.out_annotations.shape[1],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (self.out_annotations.shape[1],)),
        )

    def forward(
        self,
        x: torch.Tensor = None,
        exogenous: torch.Tensor = None,
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
        if exogenous is not None:
            logits = self.exogenous_layer(exogenous)
        elif x is not None:
            logits = self.global_layer(x)
        else:
            # raise error explaining
            raise RuntimeError()

        return logits
