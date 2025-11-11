import math
from typing import Optional, List

import torch
import numpy as np

import torch.nn.functional as F
from torch_concepts import ConceptGraph, Annotations, Variable

from ...nn.base.graph import BaseGraphLearner


class COSMOGraphLearner(BaseGraphLearner):
    def __init__(
            self,
            row_labels: List[str],
            col_labels: List[str],
            shift: float = 1.0,
            temperature: float = 1.0,
            symmetric: bool = False,
            monitor: bool = False,
            adjacency_var: float = 0.0,
            priority_var: Optional[float] = None,
            hard_threshold: bool = True,
    ):
        super(COSMOGraphLearner, self).__init__(row_labels, col_labels)

        # define COSMO parameters
        self.adj_params = torch.nn.Parameter(torch.empty((self.n_labels, self.n_labels)))
        self.np_params = torch.nn.Parameter(torch.zeros((self.n_labels, 1)))
        self.priority_var = priority_var if priority_var is not None \
            else shift / math.sqrt(2)

        # self.threshold = torch.nn.Parameter(torch.zeros(self.n_labels))
        # self.temperature = torch.nn.Parameter(torch.ones(self.n_labels) * temperature)

        self.adjacency_var = adjacency_var
        self.shift = shift
        self.temperature = temperature
        self.symmetric = symmetric
        self.monitor = monitor
        self.hard_threshold = hard_threshold
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.adj_params, nonlinearity='linear')
        torch.nn.init.normal_(self.np_params, std=self.priority_var)
        # torch.nn.init.normal_(self.threshold, std=self.priority_var)

    @property
    def orientation(self) -> torch.Tensor:
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
        # print(dif_mat)

        # Apply the shifted-tempered sigmoid
        # orient_mat = torch.sigmoid((dif_mat - self.shift) / self.temperature)
        orient_mat = dif_mat

        # Remove the diagonal
        orient_mat = orient_mat * (1 - torch.eye(n_nodes).to(orient_mat.device))

        # Hard Thresholding
        if self.hard_threshold:
            # Compute the hard orientation
            hard_orient_mat = dif_mat > self.shift
            # hard_orient_mat = dif_mat > self.threshold
            hard_orient_mat = hard_orient_mat.float()

            # Apply soft detaching trick
            orient_mat = orient_mat + (hard_orient_mat - orient_mat).detach()

        return orient_mat

    @property
    def weighted_adj(self) -> torch.Tensor:
        """
        Computes an explicit representation of the weight matrix
        given the undirected adjacency matrix and the orientation.
        """
        # orientation = self.orientation(hard_threshold=self.hard_threshold)  # nb_concepts, nb_tasks

        # Compute the adjacency matrix
        if self.symmetric:
            adj = self.adj_params + self.adj_params.T
        else:
            adj = self.adj_params

        if self.monitor:
            # Compute the weight matrix
            _weight = adj * self.orientation
            # Retain the gradient
            _weight.retain_grad()
            # Return the weight matrix
            return _weight

        return adj * self.orientation

    def forward(self):
        # compute the orientation matrix
        model_graph = self.weighted_adj
        self._model_graph = model_graph
        return model_graph

# 1 -> 5 -> 2 -> 3
# 1, 2 -> 4
# 3 -> 1