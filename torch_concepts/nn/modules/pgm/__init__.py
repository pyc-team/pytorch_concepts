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
    BayesianNetwork,
    ConceptVariable,
    EndogenousVariable,
    OpaqueVariable,
    ParametricCPD,
    ParametricFactor,
    ProbabilisticModel,
    Variable,
    param_dim,
)
from .inference import (
    AncestralInference,
    BaseInference,
    DEFAULT_GUIDES,
    DeterministicInference,
    ForwardInference,
    InferenceEngine,
    InferenceOutput,
    MVNGuide,
    NormalGuide,
    CustomGuide,
    ParamDict,
    STBernoulliGuide,
    STOneHotGuide,
    VariationalInference,
    RejectionSampling,
    dist_to_params,
    make_temperature_schedule,
)

__all__ = [
    "Variable",
    "ConceptVariable",
    "OpaqueVariable",
    "EndogenousVariable",
    "param_dim",
    "ParametricFactor",
    "ParametricCPD",
    "BayesianNetwork",
    "ProbabilisticModel",
    "InferenceOutput",
    "ParamDict",
    "BaseInference",
    "InferenceEngine",
    "ForwardInference",
    "DeterministicInference",
    "AncestralInference",
    "VariationalInference",
    "RejectionSampling",
    "DEFAULT_GUIDES",
    "CustomGuide",
    "STBernoulliGuide",
    "STOneHotGuide",
    "NormalGuide",
    "MVNGuide",
    "dist_to_params",
    "make_temperature_schedule",
]
