"""
Metrics module for concept-based model evaluation.

This module provides custom metrics for evaluating concept-based models,
including causal effect metrics and concept accuracy measures.
"""
from torchmetrics import Metric

# class ConceptCausalEffect(Metric):
#     """
#     Concept Causal Effect (CaCE) metric for measuring causal effects.
#
#     CaCE measures the causal effect between concept pairs or between a concept
#     and the task by comparing predictions under interventions do(C=1) vs do(C=0).
#
#     Note: Currently only works on binary concepts.
#
#     Attributes:
#         preds_do_1 (Tensor): Accumulated predictions under do(C=1).
#         preds_do_0 (Tensor): Accumulated predictions under do(C=0).
#         total (Tensor): Total number of samples processed.
#
#     Example:
#         >>> import torch
#         >>> from torch_concepts.nn.modules.metrics import ConceptCausalEffect
#         >>>
#         >>> # Create metric
#         >>> cace = ConceptCausalEffect()
#         >>>
#         >>> # Update with predictions under interventions
#         >>> preds_do_1 = torch.tensor([[0.1, 0.9], [0.2, 0.8]])  # P(Y|do(C=1))
#         >>> preds_do_0 = torch.tensor([[0.8, 0.2], [0.7, 0.3]])  # P(Y|do(C=0))
#         >>> cace.update(preds_do_1, preds_do_0)
#         >>>
#         >>> # Compute causal effect
#         >>> effect = cace.compute()
#         >>> print(f"Causal effect: {effect:.3f}")
#
#     References:
#         Goyal et al. "Explaining Classifiers with Causal Concept Effect (CaCE)",
#         arXiv 2019. https://arxiv.org/abs/1907.07165
#     """
#     def __init__(self):
#         super().__init__()
#         self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, 
#                preds_do_1: torch.Tensor, 
#                preds_do_0: torch.Tensor):
#         """
#         Update metric state with predictions under interventions.
#
#         Args:
#             preds_do_1: Predictions when intervening C=1, shape (batch_size, n_classes).
#             preds_do_0: Predictions when intervening C=0, shape (batch_size, n_classes).
#         """
#         _check_same_shape(preds_do_1, preds_do_0)
#         # expected value = 1*p(output=1|do(1)) + 0*(1-p(output=1|do(1))
#         self.preds_do_1 += preds_do_1[:,1].sum()
#         # expected value = 1*p(output=1|do(0)) + 0*(1-p(output=1|do(0))
#         self.preds_do_0 += preds_do_0[:,1].sum()
#         self.total += preds_do_1.size()[0]

#     def compute(self):
#         """
#         Compute the Causal Concept Effect (CaCE).
#
#         Returns:
#             torch.Tensor: The average causal effect E[Y|do(C=1)] - E[Y|do(C=0)].
#         """
#         return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float()  / self.total)
