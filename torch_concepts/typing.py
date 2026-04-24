"""Type definitions for the conceptarium package.

Provides commonly used type aliases for type hints throughout the codebase.
"""

import torch
from typing import Callable, Optional, Union

# Type alias for backbone models: can be a string (model name) or callable
# - String: 'resnet50', 'facebook/dinov2-base', etc.
# - Callable: nn.Module or any callable with signature (torch.Tensor) -> torch.Tensor
# - None: no backbone
BackboneType = Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]]