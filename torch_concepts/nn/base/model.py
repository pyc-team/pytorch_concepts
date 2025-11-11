from operator import itemgetter

import numpy as np
import torch

from torch_concepts import ConceptGraph, Annotations, nn, Variable
from typing import Union, List, Optional, Tuple

from ..modules.models.factor import Factor
from ..modules.propagator import Propagator
from .graph import BaseGraphLearner
from ...distributions import Delta


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
