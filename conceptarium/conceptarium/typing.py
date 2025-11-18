"""Type definitions for the conceptarium package.

Provides commonly used type aliases for type hints throughout the codebase.
"""

import torch
from typing import Callable, Optional

# Type alias for backbone models: callable that maps tensors to embeddings
# Can be None (no backbone), nn.Module, or any callable with the signature
# (torch.Tensor) -> torch.Tensor
BackboneType = Optional[Callable[[torch.Tensor], torch.Tensor]]