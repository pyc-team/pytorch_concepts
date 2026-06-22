"""Output containers for PGM inference engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Parameter-dict type alias
# ---------------------------------------------------------------------------

ParamDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# params -> logits assembly
# ---------------------------------------------------------------------------

# FIXME: this is a bit of a hack, but it works for now. We must make the ModelOutput
# and InferenceOutput classes more flexible in the future. Storing the tensors and 
# provide utilities for per-concept views.
def logits_from_params(
    params: Dict[str, ParamDict],
    keys: Optional[List[str]] = None,
) -> Optional[torch.Tensor]:
    """Concatenate per-variable ``'logits'`` tensors from an output's ``params``.

    The single place the library turns the per-variable parameter dict produced
    by inference into the flat ``(batch, sum_cardinalities)`` logits tensor that
    losses and metrics consume.

    Parameters
    ----------
    params : dict[str, ParamDict]
        Per-variable parameter dicts (e.g. ``{'c1': {'logits': ...}, ...}``).
    keys : list[str], optional
        Variable names to assemble, in order. When ``None`` (default), every
        variable that carries a ``'logits'`` entry is used, in insertion order.

    Returns
    -------
    torch.Tensor or None
        Concatenated logits along the last dim, or ``None`` when no queried
        variable carries logits.
    """
    if keys is None:
        keys = [n for n, p in params.items() if isinstance(p, dict) and 'logits' in p]
    parts = [params[n]['logits'] for n in keys]
    return torch.cat(parts, dim=-1) if parts else None


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



@dataclass
class ModelOutput:
    """Structured output from a high-level model's ``forward()`` method.

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
    logits: Optional[torch.Tensor] = None # FIXME: to be removed
    target: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, torch.Tensor]] = None
