import torch
from typing import Callable, Optional

BackboneType = Optional[Callable[[torch.Tensor], torch.Tensor]]