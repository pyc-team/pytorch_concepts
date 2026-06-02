from .outputs import InferenceOutput, ParamDict
from .base import BaseInference, InferenceEngine, dist_to_params, make_temperature_schedule
from .forward import ForwardInference
from .deterministic import DeterministicInference
from .ancestral import AncestralInference
from .variational import VariationalInference
from .rejection import RejectionSampling

# Re-exported for backwards compatibility; the guide classes themselves now
# live in ``torch_concepts.nn.modules.pgm.models.guides``.
from ..models.guides import (
    DEFAULT_GUIDES,
    MVNGuide,
    NormalGuide,
    CustomGuide,
    STBernoulliGuide,
    STOneHotGuide,
)

__all__ = [
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
