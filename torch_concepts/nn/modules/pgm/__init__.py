"""
Mid-level API for torch_concepts (PGM-based CBMs).

.. warning::
    Experimental APIs subject to change without a deprecation period.
"""
from .models import (
    BayesianNetwork,
    ConceptVariable,
    Delta,
    EmbeddingVariable,
    ParametricCPD,
    ParametricFactor,
    ProbabilisticModel,
    Variable,
    PARAM_DIM,
)
from .inference import (
    BaseInference,
    InferenceOutput,
    ParamDict,
    PyroBaseInference,
    PyroVariationalInference,
    TorchAncestralInference,
    TorchBaseInference,
    TorchDeterministicInference,
    TorchForwardInference,
    TorchRejectionSampling,
    dist_to_params,
    make_temperature_schedule,
)

__all__ = [
    # models
    "Variable",
    "ConceptVariable",
    "EmbeddingVariable",
    "Delta",
    "PARAM_DIM",
    "ParametricFactor",
    "ParametricCPD",
    "BayesianNetwork",
    "ProbabilisticModel",
    # inference output
    "InferenceOutput",
    "ParamDict",
    # inference base
    "BaseInference",
    # pure-torch backend
    "TorchBaseInference",
    "TorchForwardInference",
    "TorchDeterministicInference",
    "TorchAncestralInference",
    "TorchRejectionSampling",
    # pyro backend
    "PyroBaseInference",
    "PyroVariationalInference",
    # helpers
    "dist_to_params",
    "make_temperature_schedule",
]
