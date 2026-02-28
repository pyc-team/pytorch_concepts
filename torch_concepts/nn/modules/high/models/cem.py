"""Concept Embedding Model (CEM)

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
"""

from typing import List, Optional, Union

from .....annotations import Annotations

from ...low.base.inference import BaseInference
from ...low.encoders.exogenous import LinearLatentToExogenous
from ...low.encoders.linear import LinearExogenousToConcept
from ...low.predictors.exogenous import MixConceptExogegnousToConcept
from ...low.lazy import LazyConstructor

from ...mid.inference.deterministic import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel

from ..base.bipartite import BaseBipartiteModel


class ConceptEmbeddingModel(BaseBipartiteModel):
    """Concept Embedding Model with configurable training mode.
    
    A unified CEM class that works as a pure PyTorch module by default,
    or as a Lightning module when lightning=True.
    
    The CEM extends the CBM by learning concept embeddings, allowing for
    richer representations of concepts through embedding vectors.
    
    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after backbone if used).
    annotations : Annotations
        Concept annotations with labels, cardinalities, and distributions.
    task_names : Union[List[str], str]
        Names of task variables (subset of annotation labels).
    embedding_size : int, optional
        Dimensionality of concept embeddings. Defaults to 16.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
        If False (default), works as pure PyTorch module.
    inference : BaseInference, optional
        Inference engine class for evaluation. Defaults to DeterministicInference.
    train_inference : BaseInference, optional
        Inference engine class for training. Only used when lightning=True.
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
    >>> model = ConceptEmbeddingModel(
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     embedding_size=16
    ... )
    >>> out = model(x, query=['c1', 'task'])  # Direct forward pass
    
    >>> # Lightning training enabled
    >>> model = ConceptEmbeddingModel(
    ...     lightning=True,
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     embedding_size=16,
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
        embedding_size: int = 16,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = DeterministicInference,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
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

        self.eval_inference = inference(
            self.model.probabilistic_model, 
            **(inference_kwargs or {})
        )
        self.train_inference = train_inference(
            self.model.probabilistic_model, 
            **(train_inference_kwargs or {})
        )
