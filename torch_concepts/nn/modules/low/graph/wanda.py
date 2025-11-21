"""
WANDA graph learner for discovering concept relationships.

This module implements the WANDA graph
learning algorithm for discovering relations among concepts.
"""
import math
from typing import List

import torch

from ..base.graph import BaseGraphLearner


class WANDAGraphLearner(BaseGraphLearner):
    """
    WANDA Graph Learner for concept structure discovery. Adapted from COSMO.

    WANDA learns a directed acyclic graph (DAG) structure by assigning
    priority values to concepts and creating edges based on priority differences.
    This approach ensures acyclicity by construction.

    Attributes:
        np_params (nn.Parameter): Learnable priority values for each concept.
        priority_var (float): Variance for priority initialization.
        threshold (nn.Parameter): Learnable threshold for edge creation.
        hard_threshold (bool): Whether to use hard or soft thresholding.

    Args:
        row_labels: List of concept names for graph rows.
        col_labels: List of concept names for graph columns.
        priority_var: Variance for priority initialization (default: 1.0).
        hard_threshold: Use hard thresholding for edges (default: True).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import WANDAGraphLearner
        >>>
        >>> # Create WANDA learner for 5 concepts
        >>> concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
        >>> wanda = WANDAGraphLearner(
        ...     row_labels=concepts,
        ...     col_labels=concepts,
        ...     priority_var=1.0,
        ...     hard_threshold=True
        ... )
        >>>
        >>> # Get current graph estimate
        >>> adj_matrix = wanda.weighted_adj
        >>> print(adj_matrix.shape)
        torch.Size([5, 5])

    References:
        Massidda et al. "Constraint-Free Structure Learning with Smooth Acyclic
        Orientations". https://arxiv.org/abs/2309.08406
    """
    def __init__(
            self,
            row_labels: List[str],
            col_labels: List[str],
            priority_var: float = 1.0,
            hard_threshold: bool = True,
            eps: float = 1e-12,
    ):
        """
        Initialize the WANDA graph learner.

        Args:
            row_labels: List of concept names for graph rows.
            col_labels: List of concept names for graph columns.
            priority_var: Variance for priority initialization (default: 1.0).
            hard_threshold: Use hard thresholding for edges (default: True).
        """
        super(WANDAGraphLearner, self).__init__(row_labels, col_labels)

        # define COSMO parameters
        self.np_params = torch.nn.Parameter(torch.zeros((self.n_labels, 1)))
        self.priority_var = priority_var / math.sqrt(2)

        self.threshold = torch.nn.Parameter(torch.zeros(self.n_labels))

        self.eps = eps
        self.hard_threshold = hard_threshold
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset learnable parameters to initial values.

        Initializes priority parameters with normal distribution.
        """
        torch.nn.init.normal_(self.np_params, std=self.priority_var)

    @property
    def weighted_adj(self) -> torch.Tensor:
        """
        Compute the weighted adjacency matrix from learned priorities.

        Computes an orientation matrix based on priority differences. An edge
        from i to j exists when priority[j] > priority[i] + threshold[i].
        The diagonal is always zero (no self-loops).

        Returns:
            torch.Tensor: Weighted adjacency matrix of shape (n_labels, n_labels).
        """
        n_nodes = self.np_params.shape[0]

        # Difference Matrix
        dif_mat = self.np_params.T - self.np_params

        # Apply the shifted-tempered sigmoid
        orient_mat = dif_mat

        # Remove the diagonal
        orient_mat = orient_mat * (1 - torch.eye(n_nodes).to(orient_mat.device))

        # Hard Thresholding
        if self.hard_threshold:
            # Compute the hard orientation
            hard_orient_mat = dif_mat > self.threshold
            hard_orient_mat = hard_orient_mat.float()

            # Apply soft detaching trick
            zero_mat = torch.zeros_like(orient_mat)
            masked_mat = torch.where(hard_orient_mat.abs() < self.eps, zero_mat, hard_orient_mat)
            orient_mat = orient_mat + (masked_mat - orient_mat).detach()

        return orient_mat
