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

# LazyConstructor
from .modules.mid.base.model import BaseConstructor
from .modules.low.lazy import LazyConstructor

# Encoders
from .modules.low.encoders.exogenous import LinearZU
from .modules.low.encoders.linear import LinearZC, LinearUC
from .modules.low.encoders.stochastic import StochasticZC
from .modules.low.encoders.selector import SelectorZU

# Predictors
from .modules.low.predictors.linear import LinearCC
from .modules.low.predictors.exogenous import MixCUC
from .modules.low.predictors.hypernet import HyperLinearCUC
from .modules.low.predictors.call import CallableCC

# Dense layers
from .modules.low.dense_layers import Dense, ResidualMLP, MLP

# Graph learner
from .modules.low.graph.wanda import WANDAGraphLearner

# Loss functions
from .modules.loss import ConceptLoss, WeightedConceptLoss

# Metrics
from .modules.metrics import ConceptMetrics

# Models (high-level)
from .modules.high.models.blackbox import BlackBox
from .modules.high.models.cbm import ConceptBottleneckModel, \
    ConceptBottleneckModel_Joint

# Learners (high-level)
from .modules.high.learners.joint import JointLearner

# Models (mid-level)
from .modules.mid.models.cpd import ParametricCPD
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
    "BaseConstructor",

    # LazyConstructor
    "LazyConstructor",
    
    # Exogenous encoder classes
    "LinearZU",

    # Encoder classes
    "LinearZC",
    "LinearUC",
    "StochasticZC",

    # Predictor classes
    "LinearCC",
    "MixCUC",
    "HyperLinearCUC",
    "CallableCC",

    # Dense layers
    "Dense",
    "ResidualMLP",
    "MLP",

    "SelectorZU",

    # COSMO
    "WANDAGraphLearner",

    # Loss functions
    "ConceptLoss",
    "WeightedConceptLoss",

    # Metrics
    "ConceptMetrics",

    # Models (high-level)
    "BlackBox",
    # "BlackBox_torch",
    "ConceptBottleneckModel",
    "ConceptBottleneckModel_Joint",
    "ConceptBottleneckModel_Independent",

    # Learners (high-level)
    "JointLearner",

    # Models (mid-level)
    "ParametricCPD",
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
