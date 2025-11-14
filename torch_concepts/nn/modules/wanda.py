import math
from typing import Optional, List

import torch

from ...nn.base.graph import BaseGraphLearner


class WANDAGraphLearner(BaseGraphLearner):
    """
    WANDA Graph Learner Module.

    Adapted from COSMO: `"Constraint-Free Structure Learning with Smooth Acyclic Orientations" <https://arxiv.org/abs/2309.08406>`_.
    """
    def __init__(
            self,
            row_labels: List[str],
            col_labels: List[str],
            priority_var: float = 1.0,
            hard_threshold: bool = True,
    ):
        super(WANDAGraphLearner, self).__init__(row_labels, col_labels)

        # define COSMO parameters
        self.np_params = torch.nn.Parameter(torch.zeros((self.n_labels, 1)))
        self.priority_var = priority_var / math.sqrt(2)

        self.threshold = torch.nn.Parameter(torch.zeros(self.n_labels))

        self.hard_threshold = hard_threshold
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.np_params, std=self.priority_var)

    @property
    def weighted_adj(self) -> torch.Tensor:
        """
        Computes the orientation matrix given the priority vectors.
        If the hard_threshold flag is set to True, the orientation
        if thresholded against the shift parameter.

        The matrix containing the priority differences is computed
        as diff_mat[i, j] = priority[j] - priority[i]. We want an arc
        whenever p[i] < p[j], therefore, whenever
            dif_mat[i, j] > self.shift
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
            eps = 1e-12  # or smaller, depending on your precision needs
            orient_mat = orient_mat + (torch.where(hard_orient_mat.abs() < eps, torch.zeros_like(hard_orient_mat), hard_orient_mat) - orient_mat).detach()

        return orient_mat
