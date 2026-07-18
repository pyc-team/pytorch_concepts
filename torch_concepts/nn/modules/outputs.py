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

    **Uniform contract.** An engine fills only the attributes it can produce —
    a deterministic forward pass leaves ``samples`` empty, a rejection sampler
    fills only ``probabilities`` — but whenever an attribute *is* filled, its
    contents mean the same thing across every engine. Consumers can therefore
    swap engines without rewriting their losses and metrics.

    Attributes
    ----------
    params : dict[str, ParamDict]
        Per-variable distribution parameters, keyed by the variable's own
        parameter names and shaped ``(batch, *variable.shape)`` — exactly what
        the variable's distribution family takes (``{'probs': ...}`` /
        ``{'logits': ...}`` for a Bernoulli concept, ``{'loc': ..., 'scale':
        ...}`` for a Normal). Engines that compute a posterior marginal
        (e.g. :class:`~torch_concepts.nn.BeliefPropagation`) report it in this
        same parametrization rather than as a state-space belief. Only queried
        variables appear; fully-observed ones emit no parameters.
    guide_params : dict[str, ParamDict]
        Per-latent parameters of the variational guide, same layout as
        ``params``.
    samples : dict[str, torch.Tensor]
        Per-variable realisations, shaped ``(batch, *variable.shape)``. Filled
        only by engines that actually draw samples.
    probabilities : torch.Tensor or None
        ``(batch,)`` estimate of ``P(query | evidence)`` for a fully realised
        query batch. Filled only by the sampling estimators.
    """

    params: Dict[str, ParamDict] = field(default_factory=dict)
    guide_params: Dict[str, ParamDict] = field(default_factory=dict)
    samples: Dict[str, torch.Tensor] = field(default_factory=dict)
    probabilities: Optional[torch.Tensor] = None


@dataclass
class ModelOutput(InferenceOutput):
    """Structured output from a high-level model's ``forward()`` method.

    An :class:`InferenceOutput` — same four attributes, same contract — plus
    the extra fields the high level attaches around a query.

    Attributes
    ----------
    logits : torch.Tensor or None
        Flat ``(batch, sum_cardinalities)`` logits assembled from ``params``.

        .. deprecated::
           Derived state kept for the high-level losses/metrics; build it on
           demand with :func:`logits_from_params` instead.
    target : torch.Tensor or None
        The prepared ground-truth tensor aligned with ``logits``.
    extra : dict[str, torch.Tensor] or None
        Model-specific extras that do not belong to the inference contract.
    """

    logits: Optional[torch.Tensor] = None  # FIXME: to be removed
    target: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, torch.Tensor]] = None
