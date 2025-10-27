from .base.graph import BaseGraphLearner
from .base.model import BaseModel
from .base.layer import (
    BaseConceptLayer,
    BaseEncoder,
    BasePredictor,
)
from .base.inference import BaseInference, BaseIntervention

from torch_concepts.nn.modules.propagator import Propagator

from .modules.exogenous.exogenous import ExogEncoder

from .modules.encoders.linear import ProbEncoder
# from .modules.encoders.residual import LinearConceptResidualLayer
from .modules.encoders.embedding import ProbEmbEncoder
# from .modules.encoders.stochastic import StochasticConceptLayer

from .modules.predictors.linear import ProbPredictor
from .modules.predictors.embedding import MixProbEmbPredictor, HyperNetLinearPredictor

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
from .modules.inference.intervention import (
    ConstantTensorIntervention,
    ConstantLikeIntervention,
    DistributionIntervention,
    intervene_in_dict,
)


__all__ = [
    # Base classes
    "BaseConceptLayer",
    "BaseEncoder",
    "BasePredictor",
    "BaseGraphLearner",
    "BaseModel",
    "BaseInference",
    "BaseIntervention",

    # Propagator
    "Propagator",
    
    # Exogenous encoder classes
    "ExogEncoder",

    # Encoder classes
    "ProbEncoder",
    # "LinearConceptResidualLayer",
    "ProbEmbEncoder",
    # "StochasticConceptLayer",

    # Predictor classes
    "ProbPredictor",
    "MixProbEmbPredictor",
    "HyperNetLinearPredictor",

    # COSMO
    "COSMOGraphLearner",

    # Models
    "BipartiteModel",
    "GraphModel",
    "LearnedGraphModel",

    # Inference
    "KnownGraphInference",
    "UnknownGraphInference",
    # Interventions
    "ConstantTensorIntervention",
    "ConstantLikeIntervention",
    "DistributionIntervention",
    "intervene_in_dict",
]
