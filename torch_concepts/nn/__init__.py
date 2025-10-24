from .base.graph import BaseGraphLearner
from .base.model import BaseModel
from .base.layer import (
    BaseConceptLayer,
    BaseEncoderLayer,
    BasePredictorLayer,
)

from torch_concepts.nn.modules.propagator import Propagator

from .modules.encoders.linear import ProbEncoderLayer
# from .modules.encoders.residual import LinearConceptResidualLayer
from .modules.encoders.embedding import ProbEmbEncoderLayer
# from .modules.encoders.stochastic import StochasticConceptLayer

from .modules.predictors.linear import ProbPredictorLayer
from .modules.predictors.embedding import MixProbEmbPredictorLayer

from .modules.cosmo import COSMOGraphLearner

from .modules.models.bipartite import BipartiteModel
from .modules.models.graph import (
    GraphModel,
    LearnedGraphModel,
)

from .modules.inference.forward import (
    KnownGraphInference,
    UnknownGraphInference,
)


__all__ = [
    # Base classes
    "BaseConceptLayer",
    "BaseEncoderLayer",
    "BasePredictorLayer",
    "BaseGraphLearner",
    "BaseModel",

    # Propagator
    "Propagator",

    # Encoder classes
    "ProbEncoderLayer",
    # "LinearConceptResidualLayer",
    "ProbEmbEncoderLayer",
    # "StochasticConceptLayer",

    # Predictor classes
    "ProbPredictorLayer",
    "MixProbEmbPredictorLayer",

    # COSMO
    "COSMOGraphLearner",

    # Models
    "BipartiteModel",
    "GraphModel",
    "LearnedGraphModel",

    # Inference
    "KnownGraphInference",
    "UnknownGraphInference",
]
