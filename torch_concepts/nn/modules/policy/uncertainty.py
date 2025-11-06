import torch

from .... import Annotations
from ....nn.base.layer import BaseConceptLayer
from typing import List, Union, Optional


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
        subset: Optional[List[str]] = None,
    ):
        super().__init__(
            out_features=out_annotations.shape[1],
        )
        self.out_annotations = out_annotations
        self.subset = subset

    def forward(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        return logits.abs()
