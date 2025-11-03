import torch

from torch_concepts import Annotations, ConceptTensor
from ....nn.base.layer import BaseConceptLayer
from typing import List, Callable, Union, Dict, Tuple


class UncertaintyInterventionPolicy(BaseConceptLayer):
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
        out_annotations: Annotations,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_logits=None,
            in_features_embedding=None,
            in_features_exogenous=None,
            out_annotations=out_annotations,
        )

    def forward(
        self,
        logits: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return logits.abs()
