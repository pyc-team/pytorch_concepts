import torch

from torch_concepts import Annotations
from ..modules.propagator import Propagator


class BaseModel(torch.nn.Module):
    """
    BaseModel is an abstract class for all Model modules.
    """

    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,  # layer for root concepts
                 predictor: Propagator,
                 *args,
                 **kwargs,
                 ):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.annotations = annotations

        self._encoder_builder = encoder
        self._predictor_builder = predictor

        self.labels = annotations.get_axis_labels(axis=1)
        self.name2id = {name: i for i, name in enumerate(self.labels)}
