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
    OpaqueVariable,
    ParametricCPD,
    ParametricFactor,
    ProbabilisticModel,
    Variable,
    PARAM_DIM,
)
from .inference import (
    AncestralInference,
    BaseInference,
    DeterministicInference,
    ForwardInference,
    InferenceOutput,
    ParamDict,
    VariationalInference,
    RejectionSampling,
    dist_to_params,
    make_temperature_schedule,
)

__all__ = [
    "Variable",
    "ConceptVariable",
    "OpaqueVariable",
    "PARAM_DIM",
    "ParametricFactor",
    "ParametricCPD",
    "BayesianNetwork",
    "ProbabilisticModel",
    "InferenceOutput",
    "ParamDict",
    "BaseInference",
    "ForwardInference",
    "DeterministicInference",
    "AncestralInference",
    "VariationalInference",
    "RejectionSampling",
    "dist_to_params",
    "make_temperature_schedule",
]
