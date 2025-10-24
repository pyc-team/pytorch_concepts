# import numpy as np
# import torch
#
# from torch_concepts import AnnotatedTensor
# from ...base.layer import BaseConceptLayer
# from torch_concepts.nn.functional import intervene, concept_embedding_mixture
# from typing import List, Dict, Callable, Union, Tuple
#
#
# class ConceptEmbeddingLayer(BaseConceptLayer):
#     """
#     ConceptEmbeddingLayer creates supervised concept embeddings.
#     Main reference: `"Concept Embedding Models: Beyond the
#     Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_
#
#     Attributes:
#         in_features (int): Number of input features.
#         annotations (Union[List[str], int]): Concept dimensions.
#         activation (Callable): Activation function of concept scores.
#     """
#
#     def __init__(
#         self,
#         in_features: int,
#         annotations: Union[List[str], int],
#         embedding_size: int,
#         activation: Callable = torch.sigmoid,
#         *args,
#         **kwargs,
#     ):
#         annotations = [annotations, embedding_size]
#         n_concepts = (
#             len(annotations[0])
#             if isinstance(annotations[0], (list, np.ndarray))
#             else annotations[0]
#         )
#
#         super().__init__(
#             in_features=in_features,
#             annotations=annotations,
#         )
#
#         self._shape = [n_concepts, embedding_size * 2]
#         self.output_size = np.prod(self.shape())
#
#         self.activation = activation
#         self.linear = torch.nn.Sequential(
#             torch.nn.Linear(
#                 in_features,
#                 self.output_size,
#                 *args,
#                 **kwargs,
#             ),
#             torch.nn.Unflatten(-1, self.shape()),
#             torch.nn.LeakyReLU(),
#         )
#         self.concept_score_bottleneck = torch.nn.Sequential(
#             torch.nn.Linear(self.shape()[-1], 1),
#             torch.nn.Flatten(),
#         )
#
#     def predict(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Predict concept scores.
#
#         Args:
#             x (torch.Tensor): Input tensor.
#
#         Returns:
#             torch.Tensor: Predicted concept scores.
#         """
#         c_emb = self.linear(x)
#         return self.activation(self.concept_score_bottleneck(c_emb))
#
#     def intervene(
#         self,
#         x: torch.Tensor,
#         c_true: torch.Tensor = None,
#         intervention_idxs: torch.Tensor = None,
#         intervention_rate: float = 0.0,
#     ) -> torch.Tensor:
#         """
#         Intervene on concept scores.
#
#         Args:
#             x (torch.Tensor): Input tensor.
#             c_true (torch.Tensor): Ground truth concepts.
#             intervention_idxs (torch.Tensor): Boolean Tensor indicating
#                 which concepts to intervene on.
#             intervention_rate (float): Rate at which perform interventions.
#
#         Returns:
#             torch.Tensor: Intervened concept scores.
#         """
#         int_probs = torch.rand(x.shape[0], x.shape[1]) <= intervention_rate
#         int_probs = int_probs.to(x.device)
#         intervention_idxs = int_probs * intervention_idxs
#         return intervene(x, c_true, intervention_idxs)
#
#     def transform(
#         self, x: torch.Tensor, *args, **kwargs
#     ) -> Tuple[AnnotatedTensor, Dict]:
#         """
#         Transform input tensor.
#
#         Args:
#             x (torch.Tensor): Input tensor.
#
#         Returns:
#             Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
#                 dictionary with intermediate concepts tensors.
#         """
#         c_emb = self.linear(x)
#         c_pred = c_int = self.activation(self.concept_score_bottleneck(c_emb))
#         if "c_true" in kwargs:
#             c_int = self.intervene(c_pred, *args, **kwargs)
#         c_mix = concept_embedding_mixture(c_emb, c_int)
#         c_mix = self.annotate(c_mix)
#         c_int = self.annotate(c_int)
#         c_pred = self.annotate(c_pred)
#         return c_mix, dict(c_pred=c_pred, c_int=c_int)
