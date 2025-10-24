# import copy
# import torch
#
# from torch_concepts import AnnotatedTensor
# from typing import List, Dict, Callable, Union, Tuple
#
#
# class LinearConceptResidualLayer(LinearConceptLayer):
#     """
#     ConceptResidualLayer is a layer where a first set of neurons is aligned
#     with supervised concepts and a second set of neurons is free to encode
#     residual information.
#     Main reference: `"Promises and Pitfalls of Black-Box Concept Learning
#     Models" <https://arxiv.org/abs/2106.13314>`_
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
#         residual_size: int,
#         activation: Callable = torch.sigmoid,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(
#             in_features=in_features,
#             annotations=annotations,
#             activation=activation,
#             *args,
#             **kwargs,
#         )
#         self.residual = torch.nn.Sequential(
#             torch.nn.Linear(in_features, residual_size), torch.nn.LeakyReLU()
#         )
#         self.annotations_extended = list(copy.deepcopy(self.annotations))
#         self.annotations_extended[0] = list(self.annotations_extended[0])
#         self.annotations_extended[0].extend(
#             [f"residual_{i}" for i in range(residual_size)]
#         )
#         self.annotator_extended = Annotate(
#             self.annotations_extended,
#             self.annotated_axes,
#         )
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
#         c_pred = c_int = self.predict(x)
#         emb = self.residual(x)
#         if "c_true" in kwargs:
#             c_int = self.intervene(c_pred, *args, **kwargs)
#         c_int = self.annotate(c_int)
#         c_pred = self.annotate(c_pred)
#         c_new = torch.hstack((c_pred, emb))
#         c_new = self.annotator_extended(c_new)
#         return c_new, dict(c_pred=c_pred, c_int=c_int)
