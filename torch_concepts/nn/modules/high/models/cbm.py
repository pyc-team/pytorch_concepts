"""Unified Concept Bottleneck Model with multiple training modes.

This module provides a single ConceptBottleneckModel class that supports
joint and independent training through a `training` argument.
"""

from typing import List, Optional, Union

from .....annotations import Annotations

from ...low.base.inference import BaseInference
from ...low.encoders.linear import LinearLatentToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...low.lazy import LazyConstructor

from ...mid.inference.deterministic import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel

from ..base.bipartite import BaseBipartiteModel


class ConceptBottleneckModel(BaseBipartiteModel):
    """Concept Bottleneck Model with configurable training mode.
    
    A unified CBM class that works as a pure PyTorch module by default,
    or as a Lightning module when lightning=True.
    
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
    inference : BaseInference, optional
        Inference engine class for evaluation. Defaults to DeterministicInference.
    train_inference : BaseInference, optional
        Inference engine class for training.
        Defaults to DeterministicInference.
    variable_distributions : Mapping, optional
        Distribution classes for each concept if not in annotations.
    **kwargs
        Additional arguments passed to BaseBipartiteModel, including:
        
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
    
    Examples
    --------
    >>> # Pure PyTorch module (default)
    >>> model = ConceptBottleneckModel(
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task']
    ... )
    >>> out = model(x, query=['c1', 'task'])  # Direct forward pass
    
    >>> # Lightning training enabled
    >>> model = ConceptBottleneckModel(
    ...     lightning=True,
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     loss=my_loss,
    ...     optim_class=torch.optim.Adam,
    ...     optim_kwargs={'lr': 0.001}
    ... )
    """
    
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False, # wrap the Torch model with Lightning capabilities
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
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

        self.eval_inference = inference(
            self.model.probabilistic_model, 
            **(inference_kwargs or {})
        )
        _train_inference_cls = self._resolve_train_inference(inference, train_inference)
        self.train_inference = _train_inference_cls(
            self.model.probabilistic_model, 
            **(train_inference_kwargs or {})
        )
        