"""
Mid-level API for torch_concepts (PGM-based CBMs).

This package implements the architecture described in
``outputs/library_summaries/mid_level_api.md``. The legacy mid-level lives at
``torch_concepts.nn.modules.mid`` and is kept for backwards compatibility.

.. warning::
    Experimental APIs subject to change without a deprecation period.
"""
import pyro as _pyro
_pyro.settings.set(module_local_params=True)

from .models import (
    ConceptVariable,
    EndogenousVariable,
    ExogenousVariable,
    ParametricCPD,
    ParametricFactor,
    ProbabilisticModel,
    Variable,
    param_dim,
)
from .inference import (
    AncestralInference,
    DEFAULT_GUIDES,
    DeterministicInference,
    InferenceEngine,
    InferenceOutput,
    InferenceResult,
    MVNGuide,
    NormalGuide,
    ParamDict,
    STBernoulliGuide,
    STOneHotGuide,
    VariationalInference,
    dist_to_params,
    make_temperature_schedule,
)

__all__ = [
    "Variable",
    "ConceptVariable",
    "ExogenousVariable",
    "EndogenousVariable",
    "param_dim",
    "ParametricFactor",
    "ParametricCPD",
    "ProbabilisticModel",
    "InferenceOutput",
    "InferenceResult",
    "ParamDict",
    "InferenceEngine",
    "DeterministicInference",
    "AncestralInference",
    "VariationalInference",
    "DEFAULT_GUIDES",
    "STBernoulliGuide",
    "STOneHotGuide",
    "NormalGuide",
    "MVNGuide",
    "dist_to_params",
    "make_temperature_schedule",
]
