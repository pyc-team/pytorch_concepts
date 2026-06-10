"""Inference engines for PGM-based models.

The package is split by backend:

- :mod:`.base`     — backend-agnostic scaffolding (abstract ``BaseInference``).
- :mod:`.pyro`     — Pyro-backed engines (e.g. ``PyroVariationalInference``).
- :mod:`.torch`    — pure-PyTorch engines (forward, ancestral, rejection).

Outputs (:class:`InferenceOutput`, :data:`ParamDict`) and pure-torch
distribution helpers live at the package root so any backend can re-use them.
"""
from .outputs import InferenceOutput, ParamDict
from .utils import build_distribution, make_temperature_schedule
from .torch.utils import (
    build_relaxed_distribution,
    propagated_value,
    sample_from,
)
from .base import BaseInference
from .pyro import PyroBaseInference, PyroVariationalInference, dist_to_params, trace_to_params
from .torch import (
    TorchAncestralInference,
    TorchBaseInference,
    TorchDeterministicInference,
    TorchForwardInference,
    TorchRejectionSampling,
)

__all__ = [
    # outputs
    "InferenceOutput",
    "ParamDict",
    # base
    "BaseInference",
    "make_temperature_schedule",
    # pure-torch backend
    "TorchBaseInference",
    "TorchForwardInference",
    "TorchDeterministicInference",
    "TorchAncestralInference",
    "TorchRejectionSampling",
    # pyro backend
    "PyroBaseInference",
    "PyroVariationalInference",
    "dist_to_params",
    "trace_to_params",
    # pure-torch distribution helpers
    "build_distribution",
    "build_relaxed_distribution",
    "propagated_value",
    "sample_from",
]
