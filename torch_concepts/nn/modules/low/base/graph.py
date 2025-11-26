"""
Base graph learner class for concept graph discovery.

This module provides the abstract base class for learning concept graphs
from data, enabling structure discovery in concept-based models.
"""
from typing import List

import torch
import torch.nn as nn

from abc import abstractmethod, ABC


class BaseGraphLearner(nn.Module, ABC):
    """
    Abstract base class for concept graph learning modules.

    This class provides the foundation for learning the structure of concept
    graphs from data. Subclasses implement specific graph learning algorithms
    such as WANDA, NOTEARS, or other structure learning methods.

    Attributes:
        row_labels (List[str]): Labels for graph rows (source concepts).
        col_labels (List[str]): Labels for graph columns (target concepts).
        n_labels (int): Number of concepts in the graph.

    Args:
        row_labels: List of concept names for graph rows.
        col_labels: List of concept names for graph columns.

    Raises:
        AssertionError: If row_labels and col_labels have different lengths.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BaseGraphLearner
        >>>
        >>> class MyGraphLearner(BaseGraphLearner):
        ...     def __init__(self, row_labels, col_labels):
        ...         super().__init__(row_labels, col_labels)
        ...         self.graph_params = torch.nn.Parameter(
        ...             torch.randn(self.n_labels, self.n_labels)
        ...         )
        ...
        ...     def weighted_adj(self):
        ...         return torch.sigmoid(self.graph_params)
        >>>
        >>> # Create learner
        >>> concepts = ['c1', 'c2', 'c3']
        >>> learner = MyGraphLearner(concepts, concepts)
        >>> adj_matrix = learner.weighted_adj()
        >>> print(adj_matrix.shape)
        torch.Size([3, 3])
    """

    def __init__(self, row_labels: List[str], col_labels: List[str]):
        """
        Initialize the graph learner.

        Args:
            row_labels: List of concept names for graph rows.
            col_labels: List of concept names for graph columns.
        """
        super().__init__()
        assert len(row_labels) == len(col_labels)
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.n_labels = len(row_labels) # TODO: check what happens when cardinality > 1

    @abstractmethod
    def weighted_adj(self) -> torch.Tensor:
        """
        Return the learned weighted adjacency matrix.

        This method must be implemented by subclasses to return the current
        estimate of the concept graph's adjacency matrix.

        Returns:
            torch.Tensor: Weighted adjacency matrix of shape (n_labels, n_labels).

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError
