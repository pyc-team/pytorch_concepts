from .base.graph import BaseGraphLearner
from .base.model import BaseModel
from .base.layer import (
    BaseConceptLayer,
    BaseEncoderLayer,
    BasePredictorLayer,
)

from torch_concepts.nn.modules.propagator import Propagator

from .modules.encoders.linear import LinearEncoderLayer
# from .modules.encoders.embedding import ConceptEmbeddingLayer
# from .modules.encoders.residual import LinearConceptResidualLayer
# from .modules.encoders.stochastic import StochasticConceptLayer

from .modules.predictors.linear import LinearPredictorLayer

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
    "LinearEncoderLayer",
    # "LinearConceptResidualLayer",
    # "ConceptEmbeddingLayer",
    # "StochasticConceptLayer",

    # Predictor classes
    "LinearPredictorLayer",

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
