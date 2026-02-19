from typing import List, Optional, Union, Mapping
from torch import nn
import torch
from .....annotations import Annotations

from ...mid.inference.forward import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel
from ...mid.models.variable import LatentVariable, ConceptVariable
from ...mid.models.cpd import ParametricCPD
from ...mid.models.probabilistic_model import ProbabilisticModel


from ...low.encoders.linear import LinearLatentToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...low.lazy import LazyConstructor
from ...low.base.inference import BaseInference

from ..base.model import BaseModel
from ..learners import JointLearner #, IndependentLearner


class ConceptBottleneckModel_Joint(BaseModel, JointLearner):
    """High-level Concept Bottleneck Model using BipartiteModel.

    Implements a two-stage architecture:
    1. Backbone + Latent Encoder + Concept Encoder → Concept predictions
    2. Concept predictions → Task predictions

    Example:
        >>> from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel_Joint
        >>> from torch_concepts.annotations import AxisAnnotation, Annotations
        >>> from torch.distributions import Categorical, Bernoulli
        >>> ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'], 
                cardinalities=[2, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'continuous', 'distribution': Bernoulli}
                }
            )})
        >>> model = ConceptBottleneckModel_Joint(
        ...     input_size=8,
        ...     annotations=ann,
        ...     task_names=['task'],
        ...     variable_distributions=None
        ... )
        >>> x = torch.randn(2, 8)
        >>> out = model(x, query=['c1', 'task'])
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        variable_distributions: Optional[Mapping] = None,
        inference: Optional[BaseInference] = DeterministicInference,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            variable_distributions=variable_distributions,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

        self.model = BipartiteModel(
            task_names=task_names,
            input_size=self.latent_size,
            annotations=annotations,
            encoder=LazyConstructor(LinearLatentToConcept),
            predictor=LazyConstructor(LinearConceptToConcept)
        )

        self.inference = inference(self.model.probabilistic_model)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None
        ) -> torch.Tensor:
        """Forward pass through CBM.

        Args:
            x (torch.Tensor): Input data (raw or pre-computed inputs).
            query (List[str], optional): Variables to query from PGM.
                Typically all concepts and tasks. Defaults to None.
            backbone_kwargs (Optional[Mapping[str, Any]], optional): Arguments
                for backbone. Defaults to None.
            *args, **kwargs: Additional arguments for future extensions.

        Returns:
            torch.Tensor: Concatenated endogenous for queried variables.
                Shape: (batch_size, sum of variable cardinalities).
        """

        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x)

        # (b, backbone_out_features) -> (b, latent_size)
        latent = self.latent_encoder(features)

        # inference
        # get endogenous for the query concepts
        # (b, latent_size) -> (b, sum(concept_cardinalities))
        endogenous = self.inference.query(query, evidence={'input': latent})
        return endogenous

    def filter_output_for_loss(self, forward_out, target):
        """No filtering needed - return raw endogenous for standard loss computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for loss computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'input': forward_out,
                'target': target}

    def filter_output_for_metrics(self, forward_out, target):
        """No filtering needed - return raw endogenous for metric computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for metric computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'preds': forward_out,
                'target': target}


# TODO: check if mid-level implementation matches wrapping with constructor# class ConceptBottleneckModel_Joint(BaseModel, JointLearner):

#     def __init__(
#         self,
#         input_size: int,
#         annotations: Annotations,
#         task_names: Union[List[str], str],
#         variable_distributions: Optional[Mapping] = None,
#         inference: Optional[BaseInference] = DeterministicInference,
#         loss: Optional[nn.Module] = None,
#         metrics: Optional[Mapping] = None,
#         **kwargs
#     ):
#         super().__init__(
#             input_size=input_size,
#             annotations=annotations,
#             variable_distributions=variable_distributions,
#             loss=loss,
#             metrics=metrics,
#             **kwargs
#         )

#         # the initial latent embedding is provided by the BaseModel initialization
#         latent_var = LatentVariable('input', parents=[], size=self.latent_size)
#         latent_cpd = ParametricCPD('input', parametrization=nn.Identity())

#         # concepts are endogenous variables with latent as parent
#         concept_names = [var for var in self.concept_names if var not in task_names]
#         concept_dist = [self.concept_annotations.metadata[name]['distribution'] 
#                         for name in concept_names]
#         concept_size = [self.concept_annotations.cardinalities[self.concept_annotations.get_index(name)] 
#                         for name in concept_names]
#         concepts = ConceptVariable(concept_names, 
#                                       parents=["input"], 
#                                       distribution=concept_dist, 
#                                       size=concept_size)
#         concepts_cpd = ParametricCPD(concept_names, 
#                                      parametrization=LazyConstructor(LinearLatentToConcept))
        
#         # tasks are endogenous variables with concepts as parents
#         task_dist = [self.concept_annotations.metadata[name]['distribution']
#                      for name in task_names]
#         tasks = ConceptVariable(task_names, 
#                                    parents=concept_names, 
#                                    distribution=task_dist)
#         tasks_cpd = ParametricCPD(task_names, 
#                                   parametrization=LazyConstructor(LinearConceptToConcept))
        
#         # probabilistic graphical model
#         self.pgm = ProbabilisticModel(
#             variables=[latent_var, *concepts, *tasks], 
#             parametric_cpds=[latent_cpd, *concepts_cpd, *tasks_cpd]
#         )

#         self.inference = inference(self.pgm)



# TODO: implement Independent learner

# class ConceptBottleneckModel_Independent(BaseModel, IndependentLearner):
#     def __init__(
#         self,
#         input_size: int,
#         annotations: Annotations,
#         task_names: Union[List[str], str],
#         variable_distributions: Optional[Mapping] = None,
#         inference: Optional[BaseInference] = DeterministicInference,
#         loss: Optional[nn.Module] = None,
#         metrics: Optional[Mapping] = None,
#         **kwargs
#     ):
#         # Use super() for cooperative multiple inheritance
#         super().__init__(
#             input_size=input_size,
#             annotations=annotations,
#             variable_distributions=variable_distributions,
#             loss=loss,
#             metrics=metrics,
#             **kwargs
#         )
        
#         self.model = BipartiteModel(
#             task_names=task_names,
#             input_size=self.latent_size,
#             annotations=annotations,
#             encoder=LazyConstructor(LinearLatentToConcept),
#             predictor=LazyConstructor(LinearConceptToConcept)
#         )

#         self.inference = inference(self.model.probabilistic_model)

#         # Set graph_levels after model creation (deferred initialization)
#         _, graph_levels = self.inference._topological_sort()
#         graph_levels = [[var.concepts[0] for var in level] for level in graph_levels]
#         self.graph_levels = graph_levels[1:]
#         self.roots = self.graph_levels[0]
    
#     def concept_encoder(
#         self,
#         x: torch.Tensor,
#         query: List[str],
#     ) -> torch.Tensor:
#         """Forward pass through CBM.

#         Args:
#             x (torch.Tensor): Input data (raw or pre-computed inputs).
#             query (List[str], optional): Variables to query from PGM.
#                 Typically all concepts and tasks. Defaults to None.
#             backbone_kwargs (Optional[Mapping[str, Any]], optional): Arguments
#                 for backbone. Defaults to None.
#             *args, **kwargs: Additional arguments for future extensions.

#         Returns:
#             torch.Tensor: Concatenated endogenous for queried variables.
#                 Shape: (batch_size, sum of variable cardinalities).
#         """

#         # (b, input_size) -> (b, backbone_out_features)
#         features = self.maybe_apply_backbone(x)

#         # (b, backbone_out_features) -> (b, latent_size)
#         latent = self.latent_encoder(features)

#         # inference
#         # get endogenous for the query concepts
#         # (b, latent_size) -> (b, sum(concept_cardinalities))
#         endogenous = self.inference.query(query, evidence={'input': latent})
#         return endogenous

#     def concept_predictor(
#         self,
#         evidence: Mapping[str, torch.Tensor],
#         query: List[str]
#     ) -> torch.Tensor:
#         """Predict concepts from given evidence.

#         Args:
#             evidence (torch.Tensor): Evidence tensor (e.g., concept predictions).
#             query (List[str], optional): Variables to query from PGM.
#                 Typically all concepts and tasks. Defaults to None.
#             *args, **kwargs: Additional arguments for future extensions.
#         Returns:
#             torch.Tensor: Concatenated endogenous for queried variables.
#                 Shape: (batch_size, sum of variable cardinalities).
#         """
#         # inference
#         # get endogenous for the query concepts
#         # (b, evidence_size) -> (b, sum(concept_cardinalities))

#         endogenous = self.inference.query(query, evidence=evidence)
#         return endogenous

#     def filter_output_for_loss(self, forward_out, target):
#         """No filtering needed - return raw endogenous for standard loss computation.

#         Args:
#             forward_out: Model output endogenous.
#             target: Ground truth labels.

#         Returns:
#             Dict with 'input' and 'target' for loss computation.
#             This is the standard signature for pytorch Loss functions.
#         """
#         # forward_out: endogenous
#         # return: endogenous
#         return {'input': forward_out,
#                 'target': target}

#     def filter_output_for_metrics(self, forward_out, target):
#         """No filtering needed - return raw endogenous for metric computation.

#         Args:
#             forward_out: Model output endogenous.
#             target: Ground truth labels.

#         Returns:
#             Dict with 'preds' and 'target' for metric computation.
#             This is the standard signature for torchmetrics Metrics.
#         """
#         # forward_out: endogenous
#         # return: endogenous
#         return {'preds': forward_out,
#                 'target': target}
    

class ConceptBottleneckModel(ConceptBottleneckModel_Joint):
    """Alias for ConceptBottleneckModel_Joint."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)