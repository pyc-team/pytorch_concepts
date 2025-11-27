import torch
from torch import nn
from typing import List, Optional, Mapping


from .....annotations import Annotations

from ...low.dense_layers import MLP
from ..base.model import BaseModel
from ..learners import JointLearner



class BlackBox(BaseModel, JointLearner):
    """
    BlackBox model.

    This model implements a standard neural network architecture for concept-based tasks,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone for feature extraction and a latent encoder for concepts prediction.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        loss (nn.Module, optional): Loss function for training.
        metrics (Mapping, optional): Metrics for evaluation.
        backbone (nn.Module, optional): Feature extraction module.
        latent_encoder (nn.Module, optional): Latent encoder module.
        latent_encoder_kwargs (dict, optional): Arguments for latent encoder.
        **kwargs: Additional arguments for BaseModel.

    Example:
        >>> model = BlackBox(input_size=8, annotations=ann)
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=None,
            variable_distributions=None,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
    
    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
        ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x)
        endogenous = self.latent_encoder(features)
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