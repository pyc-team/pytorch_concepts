"""Base class for bipartite concept models (CBM, CEM, etc.).

This module provides a BaseBipartiteModel that handles common functionality
for models with bipartite graph structure (concepts → tasks).
"""

from typing import List, Optional, Union, Mapping, Dict
from torch import nn
import torch
from abc import abstractmethod

from .....annotations import Annotations
from .....data.utils import ensure_list
from ...mid.inference.forward import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel
from ...low.lazy import LazyConstructor
from ...low.base.inference import BaseInference
from .model import BaseModel


class BaseBipartiteModel(BaseModel):
    """Base class for bipartite concept models.
    
    Provides common functionality for models with bipartite graph structure
    where concepts predict tasks. Subclasses (CBM, CEM) only need to implement
    `__init__` to configure the specific encoder/predictor architecture.
    
    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after backbone if used).
    annotations : Annotations
        Concept annotations with labels, cardinalities, and distributions.
    task_names : Union[List[str], str]
        Names of task variables (subset of annotation labels).
    training : str, optional
        Training mode. If None (default), works as pure PyTorch module.
        Options: 'joint', 'independent', 'sequential'.
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
    
    Attributes
    ----------
    task_names : List[str]
        Task variable names.
    model : BipartiteModel
        The underlying bipartite graph model.
    inference : BaseInference
        Inference engine for forward pass.
    graph_levels : List[List[str]]
        Concept names organized by topological level.
    roots : List[str]
        Root concept names (first level).
    """
    
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        training: str = None,  # Consumed by __new__ in BaseModel
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
        
        # Store task names for later use
        self.task_names = ensure_list(task_names)
        
        # Subclasses must set self.model and call:
        #   self.inference = inference(self.model.probabilistic_model)
        #   self._init_graph_levels()

    def _init_graph_levels(self):
        """Initialize graph levels from topological sort.
        
        Filters out exogenous variables - the learner only cares about concepts.
        This is safe for both CBM (no exogenous) and CEM (has exogenous).
        
        Also identifies input-dependent exogenous variables - those that depend
        only on 'input' or other input-dependent exogenous (not on concepts).
        """
        from torch_concepts.nn.modules.mid.models.variable import ExogenousVariable, ConceptVariable
        
        _, graph_levels = self.inference._topological_sort()
        
        # Identify input-dependent exogenous: those that appear BEFORE any ConceptVariable
        # in topological order. These depend only on 'input' or other such exogenous.
        self.input_dependent_exogenous = []
        concept_seen = False
        
        for level in graph_levels[1:]:  # Skip input variable level
            for var in level:
                if isinstance(var, ConceptVariable):
                    concept_seen = True
                elif isinstance(var, ExogenousVariable) and not concept_seen:
                    self.input_dependent_exogenous.append(var.concept)
        
        # Convert Variable objects to concept names, skip input level
        # Filter out exogenous variables - learner only handles concepts
        self.graph_levels = []
        for level in graph_levels[1:]:  # Skip input variable level
            concept_vars = [var.concept for var in level 
                           if not isinstance(var, ExogenousVariable)]
            if concept_vars:  # Only add non-empty levels
                self.graph_levels.append(concept_vars)
        self.roots = self.graph_levels[0] if self.graph_levels else []
    
    def forward(
        self,
        query: List[str],
        x: torch.Tensor = None,
        evidence: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """Unified forward pass for all training modes.
        
        Parameters
        ----------
        query : List[str]
            Concept names to query.
        x : torch.Tensor, optional
            Raw input tensor. Shape: (batch_size, input_size).
            If provided, backbone and latent encoder are applied.
        evidence : Dict[str, torch.Tensor], optional
            Evidence dict mapping concept names to tensors. Defaults to empty dict.
        
        Returns
        -------
        torch.Tensor
            Concatenated predictions for queried concepts.
        """
        if evidence is None:
            evidence = {}
        
        # If x is provided, process x through backbone and latent encoder
        # and add the resulting latent representation as the 'input' of the PGM
        if x is not None:
            features = self.maybe_apply_backbone(x)
            latent = self.latent_encoder(features)
            evidence['input'] = latent
        
        return self.inference.query(query, evidence=evidence)
    
    def filter_output_for_loss(self, forward_out, target):
        """No filtering needed - return concepts for standard loss computation.

        Parameters
        ----------
        forward_out : torch.Tensor
            Model output concepts.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        dict
            Dict with 'input' and 'target' for loss computation.
        """
        return {'input': forward_out, 'target': target}

    def filter_output_for_metrics(self, forward_out, target):
        """No filtering needed - return concepts for metric computation.

        Parameters
        ----------
        forward_out : torch.Tensor
            Model output concepts.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        dict
            Dict with 'preds' and 'target' for metric computation.
        """
        return {'preds': forward_out, 'target': target}
