import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BaseEncoder
from typing import List, Callable, Union, Dict, Tuple


class ProbEncoderFromEmb(BaseEncoder):
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
        out_annotations: Annotations,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_annotations=out_annotations,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                self.out_annotations.shape[1],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (self.out_annotations.shape[1],)),
        )

    def forward(
        self,
        embedding: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.encoder(embedding)


class ProbEncoderFromExog(BaseEncoder):
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
        in_features_exogenous: int,
        out_annotations: Annotations,
        n_exogenous_per_concept: int = 1,
        *args,
        **kwargs,
    ):
        self.n_exogenous_per_concept = n_exogenous_per_concept
        in_features_exogenous = in_features_exogenous * n_exogenous_per_concept
        super().__init__(
            in_features_exogenous=in_features_exogenous,
            out_annotations=out_annotations,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_exogenous,
                1,
                *args,
                **kwargs,
            ),
            torch.nn.Flatten(),
        )

    def forward(
        self,
        exogenous: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.encoder(exogenous)
