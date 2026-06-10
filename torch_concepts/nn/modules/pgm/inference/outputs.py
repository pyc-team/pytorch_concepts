"""Output containers for PGM inference engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Parameter-dict type alias
# ---------------------------------------------------------------------------

ParamDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# InferenceOutput
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """Return value of every inference engine.

    Attributes
    ----------
    params : dict[str, ParamDict]
        Per-variable named parameter tensors of the model-side distribution
        (e.g. ``{'c': {'probs': ...}}``). 
    guide_params : dict[str, ParamDict]
        Per-latent named parameter tensors of the variational guide.
    samples : dict[str, torch.Tensor]
        Per-variable sampled values.
    probabilities : torch.Tensor or None
        Joint conditional probabilities for a fully realised query batch.
    """

    params: Dict[str, ParamDict] = field(default_factory=dict)
    guide_params: Dict[str, ParamDict] = field(default_factory=dict)
    samples: Dict[str, torch.Tensor] = field(default_factory=dict)
    probabilities: Optional[torch.Tensor] = None

    @property
    def model_params(self) -> Dict[str, ParamDict]:
        """Backwards-compatibility alias for ``params``."""
        return self.params

