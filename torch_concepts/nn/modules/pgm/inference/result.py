from dataclasses import dataclass, field
from typing import Dict

import torch

ParamDict = Dict[str, torch.Tensor]


@dataclass
class InferenceOutput:
    """Return value of every inference engine.

    Attributes
    ----------
    model_params : dict[str, dict[str, Tensor]]
        Per-variable named parameter tensors of the model-side distribution
        (e.g. ``{'c': {'probs': ...}}``). Populated for every queried variable.
    guide_params : dict[str, dict[str, Tensor]]
        Per-latent named parameter tensors of the variational guide.
        Populated only by ``VariationalInference``.
    samples : dict[str, Tensor]
        Per-variable sampled values. Populated only by ``AncestralInference``.
    """

    model_params: Dict[str, ParamDict] = field(default_factory=dict)
    guide_params: Dict[str, ParamDict] = field(default_factory=dict)
    samples: Dict[str, torch.Tensor] = field(default_factory=dict)


# Backwards-compat alias.
InferenceResult = InferenceOutput
