"""
Neural network modules for concept-based models.

This module provides neural network components for building concept-based architectures.
"""

# Base classes
from torch_concepts.nn.modules.low.base.graph import BaseGraphLearner
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.nn.modules.low.base.layer import (
    BaseConceptLayer
)
from torch_concepts.nn.modules.low.base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy
)

# LazyConstructor
from .modules.mid.base.model import BaseConstructor
from .modules.low.lazy import LazyConstructor

# Encoders
from .modules.low.encoders.linear import LinearEmbeddingToConcept

# Predictors
from .modules.low.predictors.call import CallableConceptToConcept
from .modules.low.predictors.hypernet import HyperlinearConceptEmbeddingToConcept
from .modules.low.predictors.linear import LinearConceptToConcept
from .modules.low.predictors.mix import MixConceptEmbeddingToConcept

# Dense layers
from .modules.low.dense_layers import Dense, ResidualMLP, MLP
from .modules.low.sequential import ConceptSequential

# Graph learner
from .modules.low.graph.wanda import WANDAGraphLearner

# Loss functions
from .modules.loss import ConceptLoss, WeightedConceptLoss, DepthWeightedConceptLoss, \
    L1LogitRegularizer

# Metrics
from .modules.metrics import ConceptMetrics, compute_cace

# Output containers
from .modules.outputs import ModelOutput, InferenceOutput

# Models (high-level)
from .modules.high.models.blackbox import BlackBox, BlackBoxTaskOnly
from .modules.high.models.cbm import ConceptBottleneckModel
from .modules.high.models.cem import ConceptEmbeddingModel
from .modules.high.models.c2bm import CausallyReliableConceptBottleneckModel

# Models (mid-level)
from .modules.mid.models.factor import ParametricFactor
from .modules.mid.models.cpd import ParametricCPD
from .modules.mid.models.probabilistic_model import ProbabilisticModel
from .modules.mid.constructors.bipartite import BipartiteModel
from .modules.mid.constructors.graph import GraphModel

# Inference (mid-level)
from .modules.mid.inference import (
    ForwardInference,
    DeterministicInference,
    AncestralSamplingInference,
    IndependentInference,
)

# Base intervention
from .modules.low.intervention.intervention import intervention, BaseInterventionModule, InterventionModule

# Intervention strategies
from .modules.low.intervention.strategy.ground_truth import GroundTruthIntervention
from .modules.low.intervention.strategy.do import DoIntervention
from .modules.low.intervention.strategy.distribution import DistributionIntervention
from .modules.low.intervention.strategy.positive_weights import PositiveWeightsIntervention

# Intervention policies
from .modules.low.intervention.policy.uniform import UniformPolicy
from .modules.low.intervention.policy.uncertainty import UncertaintyInterventionPolicy
from .modules.low.intervention.policy.random import RandomPolicy
from .modules.low.intervention.policy.gradient import GradientPolicy


__all__ = [
    # Base classes
    "BaseConceptLayer",
    "BaseGraphLearner",
    "BaseModel",
    "BaseConceptInterventionStrategy",
    "BaseModuleInterventionStrategy",
    "BaseInterventionPolicy",
    "BaseConstructor",
    "BaseInterventionModule",

    # LazyConstructor
    "LazyConstructor",

    # Encoder classes
    "LinearEmbeddingToConcept",

    # Predictor classes
    "LinearConceptToConcept",
    "CallableConceptToConcept",

    # Dense layers
    "Dense",
    "ResidualMLP",
    "MLP",
    "ConceptSequential",

    # COSMO
    "WANDAGraphLearner",

    # Loss functions
    "ConceptLoss",
    "WeightedConceptLoss",
    "DepthWeightedConceptLoss",
    "L1LogitRegularizer",

    # Metrics
    "ConceptMetrics",
    "compute_cace",

    # Output containers
    "ModelOutput",
    "InferenceOutput",

    # Models (high-level)
    "BlackBox",
    "BlackBoxTaskOnly",
    "ConceptBottleneckModel",
    "CausallyReliableConceptBottleneckModel",
    "ConceptEmbeddingModel",

    # Models (mid-level)
    "ParametricCPD",
    "ProbabilisticModel",
    "BipartiteModel",
    "GraphModel",

    # Inference
    "ForwardInference",
    "DeterministicInference",
    "AncestralSamplingInference",
    "IndependentInference",

    # Interventions
    "GroundTruthIntervention",
    "DoIntervention",
    "DistributionIntervention",
    "PositiveWeightsIntervention",
    "intervention",

    # Intervention policies
    "UniformPolicy",
    "UncertaintyInterventionPolicy",
    "RandomPolicy",
    "GradientPolicy",
]
