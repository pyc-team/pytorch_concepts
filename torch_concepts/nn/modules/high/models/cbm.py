from typing import Any, Dict, List, Optional, Type, Union, Mapping
from torch import nn
import torch

from .....annotations import Annotations
from .....typing import BackboneType

from ....modules.mid.constructors.bipartite import BipartiteModel
from ....modules.low.encoders.linear import ProbEncoderFromEmb
from ....modules.low.predictors.linear import ProbPredictor
from ....modules.low.lazy import LazyConstructor
from ....modules.low.base.inference import BaseInference

from ..base.model import BaseModel
from ..learners.joint import JointLearner


class ConceptBottleneckModel_Joint(BaseModel, JointLearner):
    """High-level Concept Bottleneck Model using BipartiteModel.

    Implements a two-stage architecture:
    1. Backbone + Encoder → Concept predictions
    2. Concept predictions → Task predictions
    """
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
        input_size: int,

        loss: nn.Module,
        metrics: Mapping,
        annotations: Annotations,
        variable_distributions: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,

        embs_precomputed: Optional[bool] = False,
        backbone: Optional[BackboneType] = None,
        encoder: Optional[nn.Module] = None,
        encoder_kwargs: Optional[Dict] = None,

        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        enable_summary_metrics: Optional[bool] = True,
        enable_perconcept_metrics: Optional[Union[bool, list]] = False,
        **kwargs
    ) -> None:
        # Initialize using super() to properly handle MRO
        super().__init__(
            #-- Learner args
            loss=loss,
            metrics=metrics,
            annotations=annotations,
            variable_distributions=variable_distributions,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            enable_summary_metrics=enable_summary_metrics,
            enable_perconcept_metrics=enable_perconcept_metrics,
            # -- BaseModel args
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs
        )

        model = BipartiteModel(task_names=task_names,
                               input_size=self.encoder_out_features,
                               annotations=annotations,
                               encoder=LazyConstructor(ProbEncoderFromEmb),
                               predictor=LazyConstructor(ProbPredictor))

        self.inference = inference(model.probabilistic_model)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None
        ) -> torch.Tensor:
        """Forward pass through CBM.

        Args:
            x (torch.Tensor): Input data (raw or pre-computed embeddings).
            query (List[str], optional): Variables to query from PGM.
                Typically all concepts and tasks. Defaults to None.
            backbone_kwargs (Optional[Mapping[str, Any]], optional): Arguments
                for backbone. Defaults to None.
            *args, **kwargs: Additional arguments for future extensions.

        Returns:
            torch.Tensor: Concatenated logits for queried variables.
                Shape: (batch_size, sum of variable cardinalities).
        """

        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x)

        # (b, backbone_out_features) -> (b, encoder_out_features)
        features = self.encoder(features)

        # inference
        # get logits for the query concepts
        # (b, encoder_out_features) -> (b, sum(concept_cardinalities))
        logits = self.inference.query(query, evidence={'embedding': features})
        return logits

    def filter_output_for_loss(self, forward_out, target):
        """No filtering needed - return raw logits for standard loss computation.

        Args:
            forward_out: Model output logits.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for loss computation.
        """
        # forward_out: logits
        # return: logits
        return {'input': forward_out,
                'target': target}

    def filter_output_for_metric(self, forward_out, target):
        """No filtering needed - return raw logits for metric computation.

        Args:
            forward_out: Model output logits.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for metric computation.
        """
        # forward_out: logits
        # return: logits
        return {'input': forward_out,
                'target': target}




# class ConceptBottleneckModel_Joint_factors(BaseModel):
#     """Mid-level Concept Bottleneck Model using Variables, ParametricCPDs, and PGM.

#     Provides more explicit control over the PGM structure compared to the
#     high-level CBM implementation. Useful for:
#     - Custom factor definitions
#     - Advanced PGM modifications
#     - Research on probabilistic concept models

#     The structure mirrors CBM but constructs the PGM manually:
#     embedding → concepts → tasks

#     Args:
#         task_names (Union[List[str], str, List[int]]): Task variable names/indices.
#         inference (BaseInference): Inference strategy class (uninstantiated).
#         input_size (int): Input feature dimensionality.
#         annotations (Annotations): Variable annotations.
#         variable_distributions (Mapping): Distribution types.
#         embs_precomputed (bool, optional): Skip backbone. Defaults to False.
#         backbone (Optional[callable], optional): Feature extractor. Defaults to None.
#         encoder_kwargs (Dict, optional): MLP encoder config. Defaults to None.
#         **kwargs: Reserved for future use.

#     Example:
#         >>> # More control over PGM structure
#         >>> model = CBM_factors(
#         ...     task_names=['disease'],
#         ...     inference=DeterministicInference,
#         ...     input_size=512,
#         ...     annotations=annotations,
#         ...     variable_distributions={'fever': 'binary', 'disease': 'categorical'},
#         ...     encoder_kwargs={'hidden_size': 64, 'n_layers': 1}
#         ... )
#         >>>
#         >>> # Access PGM components directly
#         >>> print(model.pgm.variables)  # [embedding, fever, cough, disease]
#         >>> print(model.pgm.factors)    # [embedding_factor, encoders, predictors]
#     """
#     def __init__(
#         self,
#         task_names: Union[List[str], str, List[int]],
#         inference: BaseInference,
#         input_size: int,
#         annotations: Annotations,
#         variable_distributions: Mapping,
#         embs_precomputed: bool = False,
#         backbone: Optional[callable] = None,
#         encoder_kwargs: Mapping = None,
#         **kwargs
#     ) -> None:
#         # Initialize the BaseModel
#         # this will setup the encoder (torch) layers and the annotations metadata
#         super().__init__(
#             annotations=annotations,
#             variable_distributions=variable_distributions,
#             # encoder params
#             input_size=input_size,
#             embs_precomputed=embs_precomputed,
#             backbone=backbone,
#             encoder_kwargs=encoder_kwargs,
#         )
#         # init variable for the latent embedding from the encoder
#         embedding = Variable("embedding", parents=[], distribution=Delta, size=self.encoder_out_features)
#         embedding_factor = ParametricCPD("embedding", parametrization=nn.Identity())

#         # variables initialization
#         concept_names = [c for c in annotations.get_axis_labels(1) if c not in task_names]
#         concepts = Variable(concept_names,
#                             parents=['embedding'], # all concepts have the same parent='embedding'
#                             distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
#                             size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
#         tasks = Variable(task_names,
#                          parents=concept_names, # all tasks have the same parents='concepts'
#                          distribution=[annotations[1].metadata[c]['distribution'] for c in task_names],
#                          size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in task_names])

#         # layers initialization
#         concept_encoders = ParametricCPD(concept_names, 
#                                   parametrization=[ProbEncoderFromEmb(in_features_embedding=embedding.size, 
#                                                                    out_features=c.size) for c in concepts])
        
#         task_predictors = ParametricCPD(task_names, 
#                                  parametrization=[ProbPredictor(in_features_logits=sum([c.size for c in concepts]), 
#                                                              out_features=t.size) for t in tasks])

#         # ProbabilisticModel Initialization
#         self.probabilistic_model = ProbabilisticModel(
#             variables=[embedding, *concepts, *tasks],
#             parametric_cpds=[embedding_factor, *concept_encoders, *task_predictors]
#         )

#         self.inference = inference(self.probabilistic_model)

#     def forward(self,
#                 x: torch.Tensor,
#                 query: List[str] = None,
#                 *args,
#                 backbone_kwargs: Optional[Mapping[str, Any]] = None,
#                 **kwargs
#             ) -> torch.Tensor:
#         """Forward pass through CBM_factors.

#         Identical behavior to CBM.forward() but uses manually constructed PGM.

#         Args:
#             x (torch.Tensor): Input data.
#             query (List[str], optional): Variables to query. Defaults to None.
#             backbone_kwargs (Optional[Mapping[str, Any]], optional): Backbone args.
#                 Defaults to None.

#         Returns:
#             torch.Tensor: Logits for queried variables.
#         """

#         # (b, input_size) -> (b, backbone_out_features)
#         features = self.maybe_apply_backbone(x, backbone_kwargs)

#         # (b, backbone_out_features) -> (b, encoder_out_features)
#         features = self.encoder(features)

#         # inference
#         # get logits for the query concepts
#         # (b, encoder_out_features) -> (b, sum(concept_cardinalities))
#         out = self.inference.query(query, evidence={'embedding': features})
#         return out

#     def filter_output_for_loss(self, forward_out):
#         """Return logits unchanged for loss computation."""
#         # forward_out: logits
#         # return: logits
#         return forward_out

#     def filter_output_for_metric(self, forward_out):
#         """Return logits unchanged for metric computation."""
#         # forward_out: logits
#         # return: logits
#         return forward_out
    




class ConceptBottleneckModel(ConceptBottleneckModel_Joint):
    """Alias for ConceptBottleneckModel_Joint for backward compatibility."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)