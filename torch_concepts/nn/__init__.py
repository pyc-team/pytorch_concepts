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
from .modules.low.lazy import LazyConstructor
from .modules.low.sequential import Sequential

# Priors (root-CPD parametrizations)
from .modules.low.priors import LearnablePrior, FixedPrior

# Encoders
from .modules.low.encoders.linear import LinearEmbeddingToConcept

# Predictors
from .modules.low.predictors.call import CallableConceptToConcept
from .modules.low.predictors.hypernet import HyperlinearConceptEmbeddingToConcept
from .modules.low.predictors.linear import LinearConceptToConcept
from .modules.low.predictors.mix import MixConceptEmbeddingToConcept

# Dense layers
from .modules.low.dense_layers import Dense, ResidualMLP, MLP, LinearEmbeddingEncoder, SelectorEmbeddingEncoder
from .modules.low.sequential import Sequential

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
from .modules.high.models.graph_cbm import GraphConceptBottleneckModel
from .modules.high.models.c2bm import CausallyReliableConceptBottleneckModel

# Models (mid-level)
from .modules.mid.models.factor import ParametricFactor
from .modules.mid.models.cpd import ParametricCPD
from .modules.mid.models.probabilistic_model import ProbabilisticModel
from .modules.mid.models.bayesian_network import BayesianNetwork
from .modules.mid.models.variable import Variable, ConceptVariable, EmbeddingVariable

# Inference (mid-level)
# base
from .modules.mid.inference.base import BaseInference
from .modules.mid.inference.torch.base import TorchBaseInference
from .modules.mid.inference.pyro.base import PyroBaseInference
# torch
from .modules.mid.inference.torch.forward import ForwardInference
from .modules.mid.inference.torch.deterministic import DeterministicInference
from .modules.mid.inference.torch.independent import IndependentInference
from .modules.mid.inference.torch.ancestral import AncestralSamplingInference
from .modules.mid.inference.torch.rejection import RejectionSampling
from .modules.mid.inference.torch.importance_sampling.importance_sampling import ImportanceSampling
from .modules.mid.inference.torch.importance_sampling.base_proposal import BaseProposal
from .modules.mid.inference.torch.importance_sampling.mutilated_network import MutilatedNetworkProposal
# pyro
from .modules.mid.inference.pyro.variational import VariationalInference
from .modules.mid.inference.pyro.importance import PyroImportanceSampling

from .modules.mid.intervention import intervention

# Base intervention
from .modules.low.intervention.intervention import BaseInterventionModule, InterventionModule

# Intervention strategies
from .modules.low.intervention.strategy.ground_truth import GroundTruthIntervention
from .modules.low.intervention.strategy.do import DoIntervention
from .modules.low.intervention.strategy.distribution import DistributionIntervention
from .modules.low.intervention.strategy.positive_weights import PositiveWeightsIntervention
from .modules.low.intervention.strategy.contrastive import ContrastiveIntervention

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
    "BaseInterventionModule",

    # LazyConstructor
    "LazyConstructor",

    # Priors
    "LearnablePrior",
    "FixedPrior",

    # Encoder classes
    "LinearEmbeddingToConcept",

    # Predictor classes
    "LinearConceptToConcept",
    "CallableConceptToConcept",
    "HyperlinearConceptEmbeddingToConcept",
    "MixConceptEmbeddingToConcept",

    # Dense layers
    "Dense",
    "ResidualMLP",
    "MLP",
    "Sequential",
    "LinearEmbeddingEncoder",
    "SelectorEmbeddingEncoder",

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
    "ConceptEmbeddingModel",
    "GraphConceptBottleneckModel",
    "CausallyReliableConceptBottleneckModel",

    # Models (mid-level)
    "ParametricFactor",
    "ParametricCPD",
    "ProbabilisticModel",
    "BayesianNetwork",
    "Variable",
    "ConceptVariable",
    "EmbeddingVariable",

    # Inference (mid-level)
    "BaseInference",
    "TorchBaseInference",
    "ForwardInference",
    "DeterministicInference",
    "AncestralSamplingInference",
    "RejectionSampling",
    "IndependentInference",
    "ImportanceSampling",
    "BaseProposal",
    "MutilatedNetworkProposal",
    "PyroBaseInference",
    "VariationalInference",
    "PyroImportanceSampling",

    # Interventions
    "GroundTruthIntervention",
    "DoIntervention",
    "DistributionIntervention",
    "PositiveWeightsIntervention",
    "ContrastiveIntervention",
    "intervention",

    # Intervention policies
    "UniformPolicy",
    "UncertaintyInterventionPolicy",
    "RandomPolicy",
    "GradientPolicy",
]
