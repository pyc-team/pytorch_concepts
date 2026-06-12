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
from torch_concepts.nn.modules.low.base.inference import BaseIntervention

# LazyConstructor
from .modules.low.lazy import LazyConstructor
from .modules.low.sequential import Sequential

# Encoders
from .modules.low.encoders.linear import LinearEmbeddingToConcept
from .modules.low.encoders.stochastic import StochasticEmbeddingToConcept

# Predictors
from .modules.low.predictors.linear import LinearConceptToConcept
from .modules.low.predictors.mix import MixConceptEmbeddingToConcept, MixSumConceptEmbeddingToConcept
from .modules.low.predictors.hypernet import HyperlinearConceptEmbeddingToConcept
from .modules.low.predictors.call import CallableConceptToConcept

# Dense layers
from .modules.low.dense_layers import Dense, ResidualMLP, MLP
from .modules.low.dense_layers import LinearEmbeddingEncoder, SelectorEmbeddingEncoder
from .modules.low.ops import SumOp, ResidualCorrectionOp

# Graph learner
from .modules.low.graph.wanda import WANDAGraphLearner

# Loss functions
from .modules.loss import ConceptLoss, WeightedConceptLoss, DepthWeightedConceptLoss, \
    L1LogitRegularizer

# Metrics
from .modules.metrics import ConceptMetrics, compute_cace

# Output containers
from .modules.outputs import ModelOutput, InferenceOutput

# # Models (high-level)
# from .modules.high.models.blackbox import BlackBox, BlackBoxTaskOnly
# from .modules.high.models.cbm import ConceptBottleneckModel
# from .modules.high.models.cem import ConceptEmbeddingModel
# from .modules.high.models.c2bm import CausallyReliableConceptBottleneckModel



# Models (mid-level)
from .modules.mid.models.factor import ParametricFactor
from .modules.mid.models.cpd import ParametricCPD
from .modules.mid.models.probabilistic_model import ProbabilisticModel
from .modules.mid.models.bayesian_network import BayesianNetwork
from .modules.mid.models.variable import Variable, ConceptVariable, EmbeddingVariable

# Inference (mid-level)
from .modules.mid.inference.base import BaseInference
from .modules.mid.inference.torch.base import TorchBaseInference
from .modules.mid.inference.torch.forward import ForwardInference
from .modules.mid.inference.torch.deterministic import DeterministicInference
from .modules.mid.inference.torch.ancestral import AncestralInference
from .modules.mid.inference.torch.rejection import RejectionSampling
from .modules.mid.inference.torch.independent import IndependentInference
from .modules.mid.inference.torch.importance_sampling.importance_sampling import ImportanceSampling
from .modules.mid.inference.torch.importance_sampling.base_proposal import BaseProposal
from .modules.mid.inference.torch.importance_sampling.mutilated_network import MutilatedNetworkProposal
from .modules.mid.inference.pyro.base import PyroBaseInference
from .modules.mid.inference.pyro.variational import VariationalInference
from .modules.mid.inference.pyro.importance import PyroImportanceSampling

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
    "BaseIntervention",

    # LazyConstructor
    "LazyConstructor",

    # Encoder classes
    "LinearEmbeddingToConcept",
    "StochasticEmbeddingToConcept",

    # Predictor classes
    "LinearConceptToConcept",
    "MixConceptEmbeddingToConcept",
    "MixSumConceptEmbeddingToConcept",
    "HyperlinearConceptEmbeddingToConcept",
    "CallableConceptToConcept",

    # Embedding encoders
    "LinearEmbeddingToEmbedding",
    "SelectorEmbeddingToEmbedding",

    # Deprecated aliases (kept until consumers migrate)
    "LinearLatentToConcept",
    "LinearExogenousToConcept",
    "StochasticLatentToConcept",
    "MixConceptExogegnousToConcept",
    "MixSumConceptExogenousToConcept",
    "HyperlinearConceptExogenousToConcept",
    
    # Dense layers
    "Dense",
    "ResidualMLP",
    "MLP",
    "LinearEmbeddingEncoder",
    "SelectorEmbeddingEncoder",

    # Ops
    "SumOp",
    "ResidualCorrectionOp",
    "Sequential",

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
    "AncestralInference",
    "RejectionSampling",
    "IndependentInference",
    "ImportanceSampling",
    "BaseProposal",
    "MutilatedNetworkProposal",
    "PyroBaseInference",
    "VariationalInference",
    "PyroImportanceSampling",

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
