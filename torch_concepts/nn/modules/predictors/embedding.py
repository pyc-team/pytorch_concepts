import numpy as np
import torch

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor
from ...base.layer import BasePredictor
from torch_concepts.nn.functional import grouped_concept_embedding_mixture
from typing import List, Dict, Callable, Union, Tuple


class MixProbExogPredictor(BasePredictor):
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
        in_features_logits: int,
        in_features_exogenous: int,
        out_annotations: Annotations,
        in_activation: Callable = torch.sigmoid,
        in_annotations: Annotations = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_exogenous=in_features_exogenous,
            out_annotations=out_annotations,
            in_activation=in_activation,
        )
        self.in_annotations = in_annotations
        if self.in_annotations is None:
            self.groups = [1] * in_features_logits
            predictor_in_features = in_features_exogenous*in_features_logits
        else:
            self.groups = list(in_annotations.get_axis_annotation(1).cardinalities)
            assert sum(self.groups) == in_features_logits
            predictor_in_features = in_features_exogenous*len(self.groups)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                predictor_in_features,
                out_annotations.shape[1],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (out_annotations.shape[1],)),
        )

    def forward(
        self,
        logits: torch.Tensor = None,
        exogenous: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        in_probs = self.in_activation(logits)
        c_mix = grouped_concept_embedding_mixture(exogenous, in_probs, groups=self.groups)
        return self.predictor(c_mix.flatten(start_dim=1))
