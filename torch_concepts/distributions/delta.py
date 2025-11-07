import torch
from torch.distributions import Distribution
from typing import List, Dict, Any, Union, Optional


class Delta(Distribution):
    arg_constraints: Dict[str, Any] = {}
    support: Optional[torch.Tensor] = None
    has_rsample = False

    def __init__(self, value: Union[List[float], torch.Tensor], validate_args=None):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)

        super().__init__(batch_shape=torch.Size([]), validate_args=validate_args)
        self._value = value.clone()

    @property
    def mean(self):
        return self._value

    def sample(self, sample_shape=torch.Size()):
        return self._value

    def rsample(self, sample_shape=torch.Size()):
        return self._value

    def log_prob(self, value):
        return torch.zeros(value.shape[:-len(self.event_shape)])

    def __repr__(self):
        return f"Delta(value_shape={self._value.shape})"
