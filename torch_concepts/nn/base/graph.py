from typing import List

import torch.nn as nn

from abc import abstractmethod, ABC

from torch_concepts import ConceptGraph, Annotations, Variable


class BaseGraphLearner(nn.Module, ABC):
    """"""

    def __init__(self, row_labels: List[str], col_labels: List[str]):
        super().__init__()
        assert len(row_labels) == len(col_labels)
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.n_labels = len(row_labels) # TODO: check what happens when cardinality > 1

    @property
    def model_graph(self) -> ConceptGraph:
        # Return the model's graph representation
        return self._model_graph

    @abstractmethod
    def forward(self, x):
        # Define the forward pass logic here
        pass
