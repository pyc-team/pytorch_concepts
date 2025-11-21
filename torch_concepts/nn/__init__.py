"""
Neural network modules for concept-based models.

This module provides neural network components for building concept-based architectures.
"""

# Base classes
from torch_concepts.nn.modules.low.base.graph import BaseGraphLearner
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.nn.modules.low.base.layer import (
    BaseConceptLayer,
    BaseEncoder,
    BasePredictor,
)
from torch_concepts.nn.modules.low.base.inference import BaseInference, BaseIntervention

# Propagator
from .modules.propagator import Propagator

# Encoders
from .modules.low.encoders.exogenous import ExogEncoder
from .modules.low.encoders.linear import ProbEncoderFromEmb, ProbEncoderFromExog
from .modules.low.encoders.stochastic import StochasticEncoderFromEmb
from .modules.low.encoders.selector import MemorySelector

# Predictors
from .modules.low.predictors.linear import ProbPredictor
from .modules.low.predictors.embedding import MixProbExogPredictor
from .modules.low.predictors.hypernet import HyperLinearPredictor

# Graph learner
from .modules.low.graph.wanda import WANDAGraphLearner

# Loss functions
from .modules.loss import ConceptLoss, WeightedConceptLoss

# Models (high-level)
from .modules.high.models.blackbox import BlackBox
from .modules.high.models.cbm import ConceptBottleneckModel, ConceptBottleneckModel_Joint

# Learners (high-level)
from .modules.high.learners.joint import JointLearner

# Models (mid-level)
from .modules.mid.models.factor import Factor
from .modules.mid.models.probabilistic_model import ProbabilisticModel
from .modules.mid.constructors.bipartite import BipartiteModel
from .modules.mid.constructors.graph import GraphModel

# Inference (mid-level)
from .modules.mid.inference.forward import (
    ForwardInference,
    DeterministicInference,
    AncestralSamplingInference,
)

# Interventions (low-level)
from .modules.low.inference.intervention import (
    RewiringIntervention,
    GroundTruthIntervention,
    DoIntervention,
    DistributionIntervention,
    intervention,
)

# Intervention policies
from .modules.low.policy.uniform import UniformPolicy
from .modules.low.policy.uncertainty import UncertaintyInterventionPolicy
from .modules.low.policy.random import RandomPolicy

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
    "ProbEncoderFromEmb",
    "ProbEncoderFromExog",
    "StochasticEncoderFromEmb",

    # Predictor classes
    "ProbPredictor",
    "MixProbExogPredictor",
    "HyperLinearPredictor",

    "MemorySelector",

    # COSMO
    "WANDAGraphLearner",

    # Loss functions
    "ConceptLoss",
    "WeightedConceptLoss",

    # Models (high-level)
    "BlackBox",
    # "BlackBox_torch",
    "ConceptBottleneckModel",
    "ConceptBottleneckModel_Joint",

    # Learners (high-level)
    "JointLearner",

    # Models (mid-level)
    "Factor",
    "ProbabilisticModel",
    "BipartiteModel",
    "GraphModel",

    # Inference
    "ForwardInference",
    "DeterministicInference",
    "AncestralSamplingInference",

    # Interventions
    "RewiringIntervention",
    "GroundTruthIntervention",
    "DoIntervention",
    "DistributionIntervention",
    "intervention",

    # Intervention policies
    "UniformPolicy",
    "UncertaintyInterventionPolicy",
    "RandomPolicy",
]
