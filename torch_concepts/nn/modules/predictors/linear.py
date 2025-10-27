import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BasePredictor
from typing import List, Callable, Union, Dict, Tuple


class ProbPredictor(BasePredictor):
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
        in_features: Union[Tuple[Dict[str, int]], Dict[str, int]],
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
                self.in_features["concept_probs"],
                self.out_features["concept_probs"],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_shapes["concept_probs"]),
        )

    @property
    def in_shapes(self) -> Dict[str, Tuple[int, ...]]:
        in_features: Tuple[Dict] = self._in_features
        if isinstance(self._in_features, dict):
            in_features = (self._in_features,)

        in_features_summary = {"concept_probs": 0}
        for c in in_features:
            if "concept_probs" not in c.keys():
                raise ValueError("Input contracts must contain 'concept_probs' key.")
            in_features_summary["concept_probs"] += c["concept_probs"]

        return {"concept_probs": (in_features_summary["concept_probs"],)}

    def predict(
        self,
        x: Union[torch.Tensor, ConceptTensor],
        *args,
        **kwargs,
    ) -> ConceptTensor:
        """
        Predict concept scores.

        Args:
            x (Union[torch.Tensor, ConceptTensor]): Input tensor.

        Returns:
            ConceptTensor: Predicted concept scores.
        """
        c_logits = self.linear(x.concept_probs)
        c_probs = self.activation(c_logits)
        return ConceptTensor(self.out_annotations, concept_probs=c_probs)
