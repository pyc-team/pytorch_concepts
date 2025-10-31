import torch.nn as nn

from abc import abstractmethod, ABC

from torch_concepts import ConceptGraph, Annotations


class BaseGraphLearner(nn.Module, ABC):
    """"""

    def __init__(self, annotations: Annotations):
        super().__init__()
        self.annotations = annotations

    @property
    def model_graph(self) -> ConceptGraph:
        # Return the model's graph representation
        return self._model_graph

    @abstractmethod
    def forward(self, x):
        # Define the forward pass logic here
        pass
