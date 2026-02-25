"""Unified Concept Embedding Model with multiple training modes.

This module provides a single ConceptEmbeddingModel class that supports
joint and independent training through a `training` argument.
"""

from typing import List, Optional, Union, Mapping
from torch import nn

from .....annotations import Annotations
from ...mid.inference.forward import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel
from ...low.encoders.exogenous import LinearLatentToExogenous
from ...low.encoders.linear import LinearExogenousToConcept
from ...low.predictors.exogenous import MixConceptExogegnousToConcept
from ...low.lazy import LazyConstructor
from ...low.base.inference import BaseInference
from ..base.bipartite import BaseBipartiteModel


class ConceptEmbeddingModel(BaseBipartiteModel):
    """Concept Embedding Model with configurable training mode.
    
    A unified CEM class that works as a pure PyTorch module by default,
    or as a Lightning module when a training mode is specified.
    
    The CEM extends the CBM by learning concept embeddings, allowing for
    richer representations of concepts through embedding vectors.
    
    Parameters
    ----------
    training : str, optional
        Training mode. If None (default), the model works as a pure PyTorch
        module callable. If specified, enables Lightning training:
        - 'joint': Train all concepts simultaneously (standard CEM).
        - 'independent': Train level-by-level with ground truth from 
          previous levels during training, cascade during validation.
    input_size : int
        Dimensionality of input features (after backbone if used).
    annotations : Annotations
        Concept annotations with labels, cardinalities, and distributions.
    task_names : Union[List[str], str]
        Names of task variables (subset of annotation labels).
    embedding_size : int, optional
        Dimensionality of concept embeddings. Defaults to 16.
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
    >>> model = ConceptEmbeddingModel(
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     embedding_size=16
    ... )
    >>> out = model(x, query=['c1', 'task'])  # Direct forward pass
    
    >>> # Lightning with joint training
    >>> model = ConceptEmbeddingModel(
    ...     training='joint',
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     embedding_size=16,
    ...     loss=my_loss
    ... )
    
    >>> # Lightning with independent training
    >>> model = ConceptEmbeddingModel(
    ...     training='independent',
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
        embedding_size: int = 16,
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
        
        # Extract concept cardinalities (excluding tasks)
        concept_idxs = [self.concept_names.index(name) for name in self.concept_names
                        if name not in self.task_names]
        cardinalities = [self.concept_annotations.cardinalities[i] for i in concept_idxs]

        # Build bipartite model architecture with embeddings
        self.model = BipartiteModel(
            task_names=task_names,
            input_size=self.latent_size,
            annotations=annotations,
            source_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=embedding_size),
            encoder=LazyConstructor(LinearExogenousToConcept),
            predictor=LazyConstructor(MixConceptExogegnousToConcept, cardinalities=cardinalities),
            use_source_exogenous=True
        )

        # Initialize inference engine and graph levels
        self.inference = inference(self.model.probabilistic_model)
        self._init_graph_levels()
