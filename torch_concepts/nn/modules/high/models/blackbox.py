import torch
from torch import nn
from typing import List, Optional, Union

from .....data.utils import ensure_list
from .....annotations import Annotations

from ...low.dense_layers import MLP
from ..base.model import BaseModel


class BlackBox(BaseModel):
    """
    BlackBox model.

    This model implements a standard neural network architecture for concept-based tasks,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone for feature extraction and a latent encoder for concepts prediction.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        lightning (bool, optional): Enable Lightning training. Default False.
        **kwargs: Additional arguments for BaseModel.

    Example:
        >>> model = BlackBox(input_size=8, annotations=ann)
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        lightning: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **kwargs
        )
        output_size = sum(self.concept_annotations.cardinalities)
        self.linear = nn.Linear(self.latent_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        query: List[str] = None,
        evidence: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the BlackBox model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        query : List[str], optional
            Concept names to query. If provided, only the columns
            corresponding to the queried concepts are returned.
        evidence : torch.Tensor, optional
            Evidence tensor (ignored for BlackBox).
        **kwargs
            Additional arguments (ignored).
        
        Returns
        -------
        torch.Tensor
            Predictions for queried concepts.
        """
        features = self.maybe_apply_backbone(x)
        endogenous = self.latent_encoder(features)
        output = self.linear(endogenous)

        if query is not None:
            output = self.concept_annotations.slice_tensor(output, query)

        return output


class BlackBoxTaskOnly(BaseModel):
    """
    BlackBox model.

    This model implements a standard neural network architecture for predicting tasks only,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone for feature extraction and a latent encoder for concepts prediction.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        task_names (Union[List[str], str]): Task names to predict.
        lightning (bool, optional): Enable Lightning training. Default False.
        **kwargs: Additional arguments for BaseModel.

    Attributes:
        task_annotations (AxisAnnotation): Sub-annotation restricted to task
            concepts only.  Use this to build ``ConceptLoss`` / ``ConceptMetrics``.
        task_concept_idx (List[int]): Concept-level column indices used to
            slice the ground-truth target tensor to match the task-only output.

    Example:
        >>> model = BlackBoxTaskOnly(input_size=8, annotations=ann, task_names=['task'])
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        lightning: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **kwargs
        )
        self.task_names = ensure_list(task_names)
        
        # Sub-annotation restricted to task concepts only
        self.task_annotations = self.concept_annotations.subset(self.task_names)

        # Concept-level indices for slicing the ground-truth target tensor
        self.task_concept_idx = [
            self.concept_annotations.get_index(name)
            for name in self.task_names
        ]

        # Logit-level output size from the task sub-annotation
        output_size = sum(self.task_annotations.cardinalities)
        self.linear = nn.Linear(self.latent_size, output_size)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
                evidence: torch.Tensor = None,
                **kwargs
        ) -> torch.Tensor:
        """Forward pass through the BlackBoxTaskOnly model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        query : List[str], optional
            Concept names to query (ignored).
            Always returns predictions for specified task_names.
        evidence : torch.Tensor, optional
            Evidence tensor (ignored).
        **kwargs
            Additional arguments (ignored).
        """
        features = self.maybe_apply_backbone(x)
        endogenous = self.latent_encoder(features)
        output = self.linear(endogenous)
        return output

    def filter_output_for_loss(self, forward_out, target):
        """Slice target to task columns to match the task-only predictions.

        Parameters
        ----------
        forward_out : torch.Tensor
            Model output containing task predictions only.
        target : torch.Tensor
            Ground truth labels (full concept-level tensor).

        Returns
        -------
        dict
            Dict with 'input' (task predictions) and 'target' (task-only
            ground-truth columns) for loss computation.
        """
        task_target = target[:, self.task_concept_idx]
        return {'input': forward_out, 'target': task_target}

    def filter_output_for_metrics(self, forward_out, target):
        """Slice target to task columns to match the task-only predictions.

        Parameters
        ----------
        forward_out : torch.Tensor
            Model output containing task predictions only.
        target : torch.Tensor
            Ground truth labels (full concept-level tensor).

        Returns
        -------
        dict
            Dict with 'preds' (task predictions) and 'target' (task-only
            ground-truth columns) for metric computation.
        """
        task_target = target[:, self.task_concept_idx]
        return {'preds': forward_out, 'target': task_target}