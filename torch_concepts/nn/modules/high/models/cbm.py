"""Unified Concept Bottleneck Model with multiple training modes.

This module provides a single ConceptBottleneckModel class that supports
joint, independent, and sequential training through a `training` argument.
"""

from typing import List, Optional, Union, Mapping
from torch import nn

from .....annotations import Annotations
from ...mid.inference.forward import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel
from ...low.encoders.linear import LinearLatentToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...low.lazy import LazyConstructor
from ...low.base.inference import BaseInference
from ..base.bipartite import BaseBipartiteModel


class ConceptBottleneckModel(BaseBipartiteModel):
    """Concept Bottleneck Model with configurable training mode.
    
    A unified CBM class that works as a pure PyTorch module by default,
    or as a Lightning module when a training mode is specified.
    
    Parameters
    ----------
    training : str, optional
        Training mode. If None (default), the model works as a pure PyTorch
        module callable. If specified, enables Lightning training:
        - 'joint': Train all concepts simultaneously (standard CBM).
        - 'independent': Train level-by-level with ground truth from 
          previous levels during training, cascade during validation.
        - 'sequential': Two-phase training - encoder first, then freeze 
          encoder and train predictor.
    input_size : int
        Dimensionality of input features (after backbone if used).
    annotations : Annotations
        Concept annotations with labels, cardinalities, and distributions.
    task_names : Union[List[str], str]
        Names of task variables (subset of annotation labels).
    variable_distributions : Mapping, optional
        Distribution classes for each concept if not in annotations.
    inference : BaseInference, optional
        Inference engine class. Defaults to DeterministicInference.
    loss : nn.Module, optional
        Loss function for Lightning training.
    metrics : Mapping, optional
        Metrics for Lightning training.
    **kwargs
        Additional arguments passed to BaseModel.
    
    Examples
    --------
    >>> # Pure PyTorch module (no training mode)
    >>> model = ConceptBottleneckModel(
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task']
    ... )
    >>> out = model(x, query=['c1', 'task'])  # Direct forward pass
    
    >>> # Lightning with joint training
    >>> model = ConceptBottleneckModel(
    ...     training='joint',
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     loss=my_loss
    ... )
    
    >>> # Lightning with sequential training
    >>> model = ConceptBottleneckModel(
    ...     training='sequential',
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task']
    ... )
    """
    
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        training: str = None,  # Consumed by __new__, included for signature
        variable_distributions: Optional[Mapping] = None,
        inference: Optional[BaseInference] = DeterministicInference,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            training=training,
            variable_distributions=variable_distributions,
            inference=inference,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        
        # Build bipartite model architecture
        self.model = BipartiteModel(
            task_names=task_names,
            input_size=self.latent_size,
            annotations=annotations,
            encoder=LazyConstructor(LinearLatentToConcept),
            predictor=LazyConstructor(LinearConceptToConcept)
        )
        
        # Initialize inference engine and graph levels
        self.inference = inference(self.model.probabilistic_model)
        self._init_graph_levels()
