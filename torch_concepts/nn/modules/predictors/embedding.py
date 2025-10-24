import numpy as np
import torch

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor
from ...base.layer import BasePredictorLayer
from torch_concepts.nn.functional import concept_embedding_mixture
from typing import List, Dict, Callable, Union, Tuple


class MixProbEmbPredictorLayer(BasePredictorLayer):
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
        in_contracts: Union[Tuple[Dict[str, int]], Dict[str, int]],
        out_annotations: Annotations,
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_contracts=in_contracts,
            out_annotations=out_annotations,
        )
        in_features = self.in_features # for linting purposes
        out_features = self.out_features
        out_shape = self.out_shape
        out_contract = self.out_contract
        in_contract = self.in_contract


        self.activation = activation

        self._internal_emb_size = in_features[1] * in_contract["concept_embs"][0]
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self._internal_emb_size,
                out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.out_shape),
        )

    @property
    def in_features(self) -> int:
        return self.in_contract["concept_embs"]

    @property
    def in_shape(self) -> Union[torch.Size, tuple]:
        return (self.in_contract["concept_embs"],)

    @property
    def in_contract(self) -> Dict[str, int]:
        _in_contracts: Tuple[Dict] = self._in_contracts
        if isinstance(self._in_contracts, dict):
            _in_contracts = (self._in_contracts,)

        n_concepts = 0
        for c in _in_contracts:
            if "concept_embs" not in c.keys():
                raise ValueError("Input contracts must contain 'concept_embs' key.")
            n_concepts += c["concept_embs"][0]

        concept_embs = (n_concepts, c["concept_embs"][1]//2) # since we use half for probs, half for embeddings
        in_contract = {"concept_embs": tuple(concept_embs)}
        return in_contract

    def predict(
        self, x: ConceptTensor, *args, **kwargs
    ) -> ConceptTensor:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_mix = concept_embedding_mixture(x.concept_embs, x.concept_probs)
        c_probs = self.activation(self.linear(c_mix.flatten(start_dim=1)))
        return ConceptTensor(self.out_annotations, concept_probs=c_probs)
