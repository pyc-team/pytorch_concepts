from typing import List, Optional, Union, Mapping
from torch import nn
import torch

from .....annotations import Annotations
from .....typing import BackboneType

from ....modules.mid.constructors.bipartite import BipartiteModel
from ....modules.low.encoders.linear import ProbEncoderFromEmb
from ....modules.low.predictors.linear import ProbPredictor
from ....modules.low.lazy import LazyConstructor
from ....modules.low.base.inference import BaseInference
from ....modules.mid.inference.forward import DeterministicInference

from ..base.model import BaseModel
from ..learners import JointLearner


class ConceptBottleneckModel_Joint(BaseModel, JointLearner):
    """High-level Concept Bottleneck Model using BipartiteModel.

    Implements a two-stage architecture:
    1. Backbone + Latent Encoder + Concept Encoder → Concept predictions
    2. Concept predictions → Task predictions
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
            encoder=LazyConstructor(ProbEncoderFromEmb),
            predictor=LazyConstructor(ProbPredictor)
        )

        self.inference = inference(self.model.probabilistic_model)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None
        ) -> torch.Tensor:
        """Forward pass through CBM.

        Args:
            x (torch.Tensor): Input data (raw or pre-computed latent codes).
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
        endogenous = self.inference.query(query, evidence={'latent': latent})
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

    def filter_output_for_metric(self, forward_out, target):
        """No filtering needed - return raw endogenous for metric computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for metric computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'input': forward_out,
                'target': target}
        

class ConceptBottleneckModel(ConceptBottleneckModel_Joint):
    """Alias for ConceptBottleneckModel_Joint for backward compatibility."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)