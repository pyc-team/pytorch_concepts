import numpy as np
import torch

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor
from ...base.layer import BasePredictor
from torch_concepts.nn.functional import concept_embedding_mixture
from typing import List, Dict, Callable, Union, Tuple


class MixProbEmbPredictor(BasePredictor):
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
        in_features = self.in_features

        self._internal_emb_size = np.prod(self.in_shapes["concept_embs"]).item()  #FIXME: when nested
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                self._internal_emb_size,
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

        n_concepts = 0
        in_features_summary = {"concept_probs": 0}
        for c in in_features:
            if "concept_embs" not in c.keys() or "concept_probs" not in c.keys():
                raise ValueError("Input contracts must contain 'concept_embs' and 'concept_probs' keys.")
            in_features_summary["concept_probs"] += c["concept_probs"]
            n_concepts += c["concept_probs"]

        # FIXME: assuming all have same emb size
        emb_dim_standard = in_features[0]["concept_embs"] // in_features[0]["concept_probs"]
        n_states = 2  # FIXME: hardcoded for now
        emb_dim_standard = emb_dim_standard // n_states

        return {"concept_probs": (in_features_summary["concept_probs"],), "concept_embs": (n_concepts, emb_dim_standard)}

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




class HyperNetLinearPredictor(BasePredictor):
    """
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
        in_features = self.in_features

        self.hypernet = torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features["concept_embs"],
                self.in_features["concept_probs"],
                *args,
                **kwargs,
            ),
            torch.nn.Flatten(),
        )

    @property
    def in_shapes(self) -> Dict[str, Tuple[int, ...]]:
        in_features: Tuple[Dict] = self._in_features
        if isinstance(self._in_features, dict):
            raise ValueError("Input contracts must be a tuple of dicts for HyperNetLinearPredictor.")

        in_features_summary = {"concept_probs": 0}
        in_emb_size = -1
        for c in in_features:
            # check that at most one dict in in_features has 'concept_embs' key
            if "concept_embs" in c.keys() and "concept_probs" not in c.keys():
                if in_emb_size != -1:
                    raise ValueError("Input contracts must contain at most one 'concept_embs' key.")
                in_emb_size = c["concept_embs"]
            else:
                if "concept_probs" not in c.keys():
                    raise ValueError("Input contracts must contain 'concept_probs' keys.")
                in_features_summary["concept_probs"] += c["concept_probs"]

        return {"concept_probs": (in_features_summary["concept_probs"],), "concept_embs": (in_emb_size,)}

    def predict(self, 
                parent_probs: ConceptTensor, 
                self_emb: ConceptTensor,
                *args,
                **kwargs
    ) -> ConceptTensor:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        weights = self.hypernet(self_emb.concept_embs)
        c_probs = self.activation(torch.einsum('bc,bc->b', parent_probs.concept_probs, weights)).unsqueeze(-1)
        return ConceptTensor(self.out_annotations, concept_probs=c_probs)
    