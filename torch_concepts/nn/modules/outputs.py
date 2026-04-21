"""Structured output containers for models and inference engines.

This module defines typed output containers that standardize communication
between inference engines, models, losses, and metrics.

    Inference (mid-level)          Model (high-level)         Training loop
    ─────────────────────────   ──────────────────────────   ─────────────────
    query() → InferenceOutput   forward() → ModelOutput      shared_step()
      .logits  (return_logits)    .logits  (return_logits)     reads .logits,
      .probs   (return_probs)     .probs   (return_probs)      .target, etc.
      .joint   (return_joint)     .joint   (return_joint)

ConceptLoss and ConceptMetrics also unpack these.
Standard torch losses and metrics never see these containers — they always
receive plain tensors extracted from the output fields.

Examples
--------
>>> # Model returns ModelOutput
>>> output = model(x=batch_x, query=['c1', 'c2', 'task'])
>>> output.probs.shape  # (batch, total_feature_dims) — activated predictions
>>>
>>> # Use with standard torch loss (access the tensor field)
>>> loss = F.binary_cross_entropy_with_logits(output.logits, targets)
>>>
>>> # ConceptLoss accepts ModelOutput directly
>>> loss = concept_loss(output)
>>>
>>> # Inference always returns InferenceOutput
>>> result = inference.query(query, evidence, return_logits=True)
>>> result.logits.shape   # concatenated logits
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class InferenceOutput:
    """Structured output from an inference engine.

    Always returned by ``ForwardInference.query()``. Which fields
    are populated is controlled by ``return_logits``, ``return_probs``,
    and ``return_joint`` parameters passed to ``query()``.

    Attributes
    ----------
    logits : torch.Tensor, optional
        Concatenated raw logits (before activation) for queried concepts.
        Populated when ``return_logits=True``.
    probs : torch.Tensor, optional
        Concatenated activated predictions for queried concepts.
        Populated when ``return_probs=True`` (default).
    joint : torch.Tensor, optional
        Joint (unnormalized) log probabilities.
        Populated when ``return_joint=True``.
    """
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    joint: Optional[torch.Tensor] = None


@dataclass
class ModelOutput:
    """Structured output from a high-level model's ``forward()`` method.

    Which prediction fields are populated mirrors the ``return_*``
    parameters passed to ``forward()``.

    Attributes
    ----------
    logits : torch.Tensor, optional
        Concatenated logits for all queried concepts.
        Populated when ``return_logits=True``.
    probs : torch.Tensor, optional
        Concatenated activated predictions for all queried concepts.
        Populated when ``return_probs=True`` (default).
    joint : torch.Tensor, optional
        Joint (unnormalized) log probabilities.
        Populated when ``return_joint=True``.
    target : torch.Tensor, optional
        Ground truth labels. Attached by the training loop
        (``shared_step``) before passing to loss/metrics.
    extras : dict of str → torch.Tensor, optional
        Model-specific extra outputs (e.g. embeddings, latent
        representations, KL divergence terms). Loss terms can
        declare these in their ``forward()`` signature to receive
        them via signature-based dispatch.

    Examples
    --------
    >>> # Inference mode — probs populated by default
    >>> output = model(x=batch_x, query=['c1', 'c2', 'task'])
    >>> output.probs.shape  # (batch, total_logit_dims)
    >>>
    >>> # Training mode — logits for loss
    >>> output = model(x=batch_x, query=['c1', 'c2', 'task'], return_logits=True)
    >>> output.logits.shape
    >>>
    >>> # Training loop attaches target
    >>> output.target = ground_truth_tensor
    """
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    joint: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    extras: Optional[Dict[str, torch.Tensor]] = None
