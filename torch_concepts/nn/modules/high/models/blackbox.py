import torch
from torch import nn
from typing import Any, List, Optional, Dict, Mapping, Type, Union


from .....annotations import Annotations
from .....typing import BackboneType

from ...low.dense_layers import MLP
from ..base.model import BaseModel
from ..learners import JointLearner



class BlackBox(BaseModel, JointLearner):
    def __init__(
        self,
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
        summary_metrics: Optional[bool] = True,
        perconcept_metrics: Optional[Union[bool, list]] = False,
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
            summary_metrics=summary_metrics,
            perconcept_metrics=perconcept_metrics,
            #-- BaseModel args
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs
        )

        self.concept_annotations = annotations.get_axis_annotation(1)
        self.mlp = MLP(input_size=input_size,
                       output_size=sum(self.concept_annotations.cardinalities),
                       **encoder_kwargs
                       )
    
    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
        ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x)
        logits = self.mlp(features)
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