from dataclasses import dataclass, field
from typing import Dict

import torch

ParamDict = Dict[str, torch.Tensor]


@dataclass
class InferenceOutput:
    """Container for inference engine outputs (§5.1)."""

    model_params: Dict[str, ParamDict] = field(default_factory=dict)
    guide_params: Dict[str, ParamDict] = field(default_factory=dict)
    samples: Dict[str, torch.Tensor] = field(default_factory=dict)


# Backwards-compat alias.
InferenceResult = InferenceOutput
