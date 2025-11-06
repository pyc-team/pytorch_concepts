import torch
from torch.distributions import Distribution
from typing import List, Dict, Any, Union, Optional


class Delta(Distribution):
    arg_constraints: Dict[str, Any] = {}
    support: Optional[torch.Tensor] = None
    has_rsample = False

    def __init__(self, size: int, value: Union[List[float], torch.Tensor], validate_args=None):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)

        if value.shape != torch.Size([size]):
            if size == 1 and value.shape == torch.Size([]):
                value = value.unsqueeze(0)
            elif value.shape == torch.Size([1]):
                value = value.repeat(size)
            else:
                raise ValueError(f"Value shape {value.shape} must match size {size}. Got: {value.tolist()}")

        super().__init__(batch_shape=torch.Size([]), event_shape=torch.Size([size]), validate_args=validate_args)
        self.size = size
        self._value = value.clone()

    @property
    def mean(self):
        return self._value

    def sample(self, sample_shape=torch.Size()):
        return self._value.expand(sample_shape + self.event_shape)

    def log_prob(self, value):
        return torch.zeros(value.shape[:-len(self.event_shape)])

    def __repr__(self):
        return f"Delta(size={self.size}, value_shape={self._value.shape})"
