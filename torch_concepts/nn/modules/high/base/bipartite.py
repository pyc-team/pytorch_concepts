"""Base class for bipartite concept models (CBM, CEM, etc.).

This module provides a BaseBipartiteModel that handles common functionality
for models with bipartite graph structure (concepts → tasks).
"""

from typing import List, Union, Dict
import torch

from .....annotations import Annotations
from .....data.utils import ensure_list
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
    lightning : bool, default False
        If True, adds Lightning training capabilities.
        If False (default), works as pure PyTorch module.
    **kwargs
        Additional arguments passed to BaseModel, including:
        
        - **backbone** : Feature extraction module (e.g., ResNet)
        - **latent_encoder** : Custom encoder for latent space
        - **latent_encoder_kwargs** : Arguments for latent encoder
        
        Lightning Training (when lightning=True):
        
        - **loss** : Loss function (nn.Module)
        - **metrics** : ConceptMetrics or dict of MetricCollections
        - **optim_class** : Optimizer class (e.g., torch.optim.Adam)
        - **optim_kwargs** : Optimizer arguments (e.g., {'lr': 0.001})
        - **scheduler_class** : LR scheduler class
        - **scheduler_kwargs** : Scheduler arguments
    
    Attributes
    ----------
    task_names : List[str]
        Task variable names.
    model : BipartiteModel
        The underlying bipartite graph model.
    inference : BaseInference
        Active inference engine (property). Returns ``train_inference``
        when in train mode, ``eval_inference`` when in eval mode.
    eval_inference : BaseInference
        Inference engine used during evaluation.
    train_inference : BaseInference
        Inference engine used during training (may be None).
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
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            **kwargs
        )
        
        # Store task names for later use
        self.task_names = ensure_list(task_names)
