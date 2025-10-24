from abc import abstractmethod

import torch

from torch_concepts import ConceptTensor


class BaseInference(torch.nn.Module):
    """
    BaseInference is an abstract class for inference modules.
    """
    def __init__(self, model: torch.nn.Module):
        super(BaseInference, self).__init__()
        self.model = model

    def forward(self,
                x: torch.Tensor,
                *args,
                **kwargs) -> ConceptTensor:
        return self.query(x, *args, **kwargs)

    @abstractmethod
    def query(self,
              x: torch.Tensor,
              c: torch.Tensor,
              *args,
              **kwargs) -> ConceptTensor:
        """
        Query model to get concepts.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Concept tensor for interventions. Defaults to None.

        Returns:
            ConceptTensor: Queried concepts.
        """
        raise NotImplementedError