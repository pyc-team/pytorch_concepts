"""Base model class for concept-based neural networks.

This module defines the abstract BaseModel class that serves as the foundation
for all concept-based models in the library. It handles backbone integration,
encoder setup, and provides hooks for data preprocessing.

BaseModel supports two usage modes:

1. **Standard PyTorch Module** (training=False, default):
   - Works as a regular nn.Module
   - Manually define optimizer, loss function, training loop
   - Full control over forward pass, loss computation, optimization
   - Ideal for custom training procedures

2. **PyTorch Lightning Module** (lightning=True):
   - Initialize model with loss, optim_class, optim_kwargs parameters
   - Use Lightning Trainer for automatic training/validation/testing
   - Inherits training logic from BaseLearner
   - Ideal for rapid experimentation with standard procedures

See Also
--------
torch_concepts.nn.modules.high.base.learner.BaseLearner : Lightning training logic
torch_concepts.nn.modules.high.models.cbm.ConceptBottleneckModel : Concrete implementation
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Mapping, Dict
import torch
import torch.nn as nn

from .....annotations import Annotations
from ...low.dense_layers import MLP
from .....typing import BackboneType
from .....utils import add_distribution_to_annotations, add_activation_to_annotations, add_default_properties
from ...utils import with_training_mode
from ...outputs import ModelOutput

from ...mid.constructors.concept_graph import ConceptGraph

class BaseModel(nn.Module, ABC):
    """Abstract base class for concept-based models.

    Provides common functionality for models that use backbones for feature extraction, 
    and encoders for latent representations. All concrete model implementations 
    should inherit from this class.

    BaseModel supports two usage modes controlled by the `lightning` parameter:

    **Mode 1: Standard PyTorch Module (lightning=False, default)**
    
    Initialize model without lightning=True for full manual control.
    You define the training loop, optimizer, and loss function externally.
    
    **Mode 2: PyTorch Lightning Module (lightning=True)**
    
    Initialize model with lightning=True, loss, optim_class, and optim_kwargs 
    for automatic training via PyTorch Lightning Trainer.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features after backbone processing. If no backbone
        is used (backbone=None), this should match raw input dimensionality.
    annotations : Annotations
        Concept annotations containing variable names, cardinalities, and optional
        distribution metadata. Distributions specify how the model represents each
        concept (e.g., Bernoulli for binary, OneHotCategorical for multi-class).
    lightning : bool, default False
        If True, adds Lightning training capabilities (BaseLearner mixin).
        If False, works as a pure PyTorch module.
    variable_distributions : Mapping, optional
        Dictionary mapping concept names to torch.distributions classes (e.g.,
        ``{'c1': Bernoulli, 'c2': OneHotCategorical}``). If None, default distributions
        are used (e.g., ``Bernoulli`` for binary, ``OneHotCategorical`` for categorical concepts).
        If provided, distributions are added to annotations internally. 
        Can also be a GroupConfig object. Defaults to None.
    variable_activations : Mapping, optional
        Dictionary mapping concept names to activation functions (e.g.,
        ``{'c1': torch.sigmoid, 'c2': torch.softmax}``). If None, default activations
        are used (e.g., ``torch.sigmoid`` for binary, ``torch.softmax`` for categorical concepts).
        If provided, activations are added to annotations internally. 
        Can also be a GroupConfig object. Defaults to None.
    graph : ConceptGraph, optional
        Directed acyclic graph (DAG) specifying causal or dependency relationships
        between concepts. Nodes correspond to concept names in annotations; edges
        encode parent-child dependencies used by graph-aware models (e.g., 
        ``CausallyReliableConceptBottleneckModel``). If None, model assumes no explicit 
        graph structure, and each model enforces its own.
        Defaults to None.
    backbone : BackboneType, optional
        Feature extraction module (e.g., ResNet, ViT) applied before latent encoder.
        Can be nn.Module or callable. If None, assumes inputs are pre-computed features.
        Defaults to None.
    latent_encoder : nn.Module, optional
        Custom encoder mapping backbone outputs to latent space. If provided,
        latent_encoder_kwargs are passed to this constructor. If None and
        latent_encoder_kwargs provided, uses MLP. Defaults to None.
    latent_encoder_kwargs : Dict, optional
        Arguments for latent encoder construction. Common keys:
        - 'hidden_size' (int): Latent dimension
        - 'n_layers' (int): Number of hidden layers
        - 'activation' (str): Activation function name
        If None, uses nn.Identity (no encoding). Defaults to None.
    
    Lightning Training Parameters (only used when lightning=True)
    -------------------------------------------------------------
    loss : nn.Module, optional
        Loss function for training (e.g. ``ConceptLoss``).  Use per-type
        composition via ``ConceptLoss`` to combine multiple terms (see
        ``binary``, ``binary_weights``, etc.).
    metrics : ConceptMetrics or dict, optional
        Metrics for evaluation. Can be a ConceptMetrics object or dict with keys
        'train_metrics', 'val_metrics', 'test_metrics' mapping to MetricCollections.
    optim_class : torch.optim.Optimizer, optional
        Optimizer class (not instance). E.g., torch.optim.Adam, torch.optim.AdamW.
    optim_kwargs : dict, optional
        Keyword arguments for optimizer. E.g., {'lr': 0.001, 'weight_decay': 1e-4}.
    scheduler_class : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler class. E.g., StepLR, CosineAnnealingLR.
    scheduler_kwargs : dict, optional
        Keyword arguments for scheduler. Include 'monitor' key for ReduceLROnPlateau.
    **kwargs
        Additional arguments passed to nn.Module superclass.

    Attributes
    ----------
    concept_annotations : AxisAnnotation
        Axis-1 annotations with distribution metadata for each concept.
    concept_names : List[str]
        List of concept variable names from annotations.
    backbone : BackboneType or None
        Feature extraction module (None if using pre-computed features).
    latent_encoder : nn.Module
        Encoder transforming backbone outputs to latent representations.
    latent_size : int
        Dimensionality of latent encoder output (input to concept encoders).

    Notes
    -----
    - **Concept Distributions and Activations**: The model needs to know which
      distribution and activation to use for each concept. These can be provided 
      in three ways:

      1. In annotations metadata before model init
      2. Via the `variable_distributions` and `variable_activations` parameters
      3. If missing, the model will fill in defaults
      If no default can be determined, a ``ValueError`` is raised.

    - Subclasses must implement ``forward()``.
    - For Lightning training, set lightning=True. The BaseLearner mixin is
      automatically added via ``__new__``.
    - The latent_size attribute is critical for downstream concept encoders
      to determine input dimensionality.

    Examples
    --------
    Distributions and activations should be in annotations metadata. If not
    provided, defaults are used (Bernoulli for binary, OneHotCategorical for
    categorical concepts):
    
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.distributions import Bernoulli
    >>> from torch_concepts.nn import ConceptBottleneckModel
    >>> from torch_concepts.annotations import AxisAnnotation, Annotations
    >>> from torch_concepts.utils import add_distribution_to_annotations
    >>> 
    >>> # Option 1: Explicit distributions in annotations metadata
    >>> ann = Annotations({
    ...     1: AxisAnnotation(
    ...         labels=['c1', 'c2', 'task'],
    ...         cardinalities=[1, 1, 1],
    ...         metadata={
    ...             'c1': {'type': 'discrete', 'distribution': Bernoulli},
    ...             'c2': {'type': 'discrete', 'distribution': Bernoulli},
    ...             'task': {'type': 'discrete', 'distribution': Bernoulli}
    ...         }
    ...     )
    ... })
    >>> model = ConceptBottleneckModel(
    ...     input_size=10,
    ...     annotations=ann, # distributions provided in metadata
    ...     task_names=['task']
    ... )
    >>> 
    >>> # Option 2: Add distributions via utility before model init
    >>> ann_no_dist = Annotations({
    ...     1: AxisAnnotation(
    ...         labels=['c1', 'c2', 'task'],
    ...         cardinalities=[1, 1, 1],
    ...         metadata={
    ...             'c1': {'type': 'discrete'},
    ...             'c2': {'type': 'discrete'},
    ...             'task': {'type': 'discrete'}
    ...         }
    ...     )
    ... })
    >>> distributions = {'c1': Bernoulli, 'c2': Bernoulli, 'task': Bernoulli}
    >>> ann_no_dist = add_distribution_to_annotations(
    ...     ann_no_dist, distributions
    ... )
    >>> model = ConceptBottleneckModel(
    ...     input_size=10,
    ...     annotations=ann_no_dist,
    ...     task_names=['task']
    ... )
    >>> 
    >>> # Option 3: Let the model use defaults (Bernoulli for binary discrete)
    >>> model = ConceptBottleneckModel(
    ...     input_size=10,
    ...     annotations=ann_no_dist,
    ...     task_names=['task']
    ... )
    >>> 
    >>> # Manual training loop
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    >>> loss_fn = nn.BCEWithLogitsLoss()
    >>> x = torch.randn(32, 10)
    >>> y = torch.randint(0, 2, (32, 3)).float()
    >>> 
    >>> for epoch in range(100):
    ...     optimizer.zero_grad()
    ...     out = model(x, query=['c1', 'c2', 'task'])
    ...     loss = loss_fn(out, y)
    ...     loss.backward()
    ...     optimizer.step()

    See Also
    --------
    torch_concepts.nn.modules.high.models.cbm.ConceptBottleneckModel : Concrete CBM implementation
    torch_concepts.nn.modules.high.base.learner.BaseLearner : Lightning training logic
    torch_concepts.annotations.Annotations : Concept annotation container
    """

    def __new__(cls, *args, lightning: bool = False, **kwargs):
        """Create instance with BaseLearner mixin for Lightning training.
        
        This method dynamically creates a combined class that includes
        BaseLearner when lightning=True.
        
        Parameters
        ----------
        lightning : bool, default False
            If True, adds BaseLearner mixin for Lightning training.
            If False, returns a pure PyTorch module without Lightning integration.
        
        Returns
        -------
        BaseModel
            Instance of the combined class with BaseLearner mixin.
        """
        combined_class = with_training_mode(cls, lightning)
        instance = object.__new__(combined_class)
        instance._lightning_enabled = lightning
        return instance

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Optional[Mapping] = None,
        variable_activations: Optional[Mapping] = None,
        graph: ConceptGraph = None,
        backbone: Optional[BackboneType] = None,
        latent_encoder: Optional[nn.Module] = None,
        latent_encoder_kwargs: Optional[Dict] = None,
        lightning: bool = False,  # Consumed by __new__, included for signature
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.graph = graph

        if annotations is not None:
            annotations = annotations.get_axis_annotation(1)

            # 1. If distributions/activations are explicitly passed, override annotations
            if variable_distributions is not None:
                annotations = add_distribution_to_annotations(annotations, variable_distributions)
            if variable_activations is not None:
                annotations = add_activation_to_annotations(annotations, variable_activations)

            # 2. Fill in defaults for any concepts still missing distribution/activation
            # this also serves as a validation step to ensure all concepts have necessary metadata
            self.concept_annotations = add_default_properties(annotations)

            self.concept_names = self.concept_annotations.labels

        self._backbone = backbone

        if latent_encoder is not None:
            self._latent_encoder = latent_encoder(
                input_size,
                **(latent_encoder_kwargs or {})
            )
        elif latent_encoder_kwargs is not None:
            # assume an MLP encoder if latent_encoder_kwargs provided but no latent_encoder
            self._latent_encoder = MLP(
                input_size=input_size,
                **latent_encoder_kwargs
            )
        else:
            self._latent_encoder = nn.Identity()

        self.latent_size = latent_encoder_kwargs.get('hidden_size') if latent_encoder_kwargs else input_size

    @property
    def inference(self):
        """Return the active inference engine based on train/eval mode.

        When ``self.training`` is True (after ``.train()``), returns
        ``self.train_inference``.  When False (after ``.eval()``),
        returns ``self.eval_inference``.  This mirrors PyTorch and
        Lightning conventions so that calling ``.train()`` / ``.eval()``
        automatically selects the correct engine.

        Returns
        -------
        BaseInference
            The currently active inference engine.
        """
        if self.training and self.train_inference is not None:
            return self.train_inference
        return self.eval_inference

    @staticmethod
    def _resolve_train_inference(inference, train_inference):
        """Validate and resolve the train_inference class.

        If ``train_inference`` is ``None`` it falls back to ``inference``.
        If it is explicitly set to a *different* class, a ``ValueError`` is
        raised because mixing inference engines for training and evaluation
        is not supported.

        Parameters
        ----------
        inference : type
            The evaluation inference class.
        train_inference : type or None
            The training inference class, or ``None`` to fall back to
            ``inference``.

        Returns
        -------
        type
            Resolved training inference class (always ``inference`` or the
            same class as ``inference``).

        Raises
        ------
        ValueError
            If ``train_inference`` is explicitly set to a different class than
            ``inference``.
        """
        if train_inference is not None and train_inference is not inference:
            raise ValueError(
                f"train_inference ({train_inference.__name__}) must be the same "
                f"class as inference ({inference.__name__}). Different inference "
                "engines for training and evaluation are not yet supported."
            )
        return train_inference if train_inference is not None else inference

    def _finalize(self):
        if not hasattr(self, 'model') or self.model is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set self.model in __init__"
            )
        if not hasattr(self, 'eval_inference') or self.eval_inference is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set self.eval_inference in __init__"
            )
        if self._lightning_enabled and not hasattr(self, 'train_inference'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must set self.train_inference in __init__ when lightning=True"
            )

    def __repr__(self):
        backbone_name = self.backbone.__class__.__name__ if self.backbone is not None else "None"
        latent_encoder_name = self._latent_encoder.__class__.__name__ if self._latent_encoder is not None else "None"
        return f"{self.__class__.__name__}(backbone={backbone_name}, latent_encoder={latent_encoder_name})"

    @property
    def backbone(self) -> BackboneType:
        """The backbone feature extractor.

        Returns the backbone module used for feature extraction from raw inputs.
        If None, the model expects pre-computed features as inputs.

        Returns
        -------
        BackboneType or None
            Backbone module (e.g., ResNet, ViT) or None if using pre-computed features.
        """
        return self._backbone

    @property
    def latent_encoder(self) -> nn.Module:
        """The encoder mapping backbone output to latent space.

        Returns the latent encoder module that transforms backbone features
        (or raw inputs if no backbone) into latent representations used by
        concept encoders.

        Returns
        -------
        nn.Module
            Latent encoder network (MLP, custom module, or nn.Identity if no encoding).
        """
        return self._latent_encoder

    # TODO: add decoder?
    # @property
    # def encoder(self) -> nn.Module:
    #     """The decoder mapping back to the input space.

    #     Returns:
    #         nn.Module: Decoder network.
    #     """
    #     return self._encoder

    def forward(
        self,
        query: List[str],
        x: torch.Tensor = None,
        evidence: Dict[str, torch.Tensor] = None,
        *inference_args,
        **inference_kwargs
    ) -> ModelOutput:
        """Unified forward pass for all inferences.

        The active inference engine is selected automatically based on
        ``self.training`` (toggled by ``.train()`` / ``.eval()``).
        
        Parameters
        ----------
        query : List[str]
            Concept names to query.
        x : torch.Tensor, optional
            Raw input tensor. Shape: (batch_size, input_size).
            If provided, backbone and latent encoder are applied.
        evidence : Dict[str, torch.Tensor], optional
            Evidence dict mapping names to tensors. Defaults to empty dict.
            Names should match variable names in the PGM.
        *inference_args
            Positional arguments passed to the inference engine's query method.
        **inference_kwargs
            Keyword arguments passed to the inference engine's query method.
            Includes ``return_logits``, ``return_probs``, ``return_joint``.
        
        Returns
        -------
        ModelOutput
            Structured output with ``.logits`` and/or ``.probs``
            populated according to ``return_logits``/``return_probs``
            in inference_kwargs.
        """
        if evidence is None:
            evidence = {}
        
        # If x is provided, process x through backbone and latent encoder
        # and add the resulting latent representation as the 'input' of the PGM
        # TODO: handle backbone kwargs when present
        if x is not None:
            features = self.maybe_apply_backbone(x)
            latent = self.latent_encoder(features)
            evidence['input'] = latent
        
        result = self.inference.query(
            query, 
            evidence=evidence,
            *inference_args, 
            **inference_kwargs
        )
        
        return ModelOutput(
            logits=result.logits,
            probs=result.probs,
            joint=result.joint,
        )

    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        """Prepare ground truth labels for loss/metrics.

        Override in subclasses that need to transform the target
        (e.g. slice to task-only columns).

        Parameters
        ----------
        target : torch.Tensor
            Raw ground truth labels from the batch.

        Returns
        -------
        torch.Tensor
            Transformed target tensor.
        """
        return target

    # ------------------------------------------------------------------
    # Features extraction helpers
    # ------------------------------------------------------------------

    def maybe_apply_backbone(
        self,
        x: torch.Tensor,
        backbone_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the backbone to ``x`` unless features are pre-computed.

        Args:
            x (torch.Tensor): Raw input tensor or already computed embeddings.
            backbone_kwargs (Any): Extra keyword arguments forwarded to the
                backbone callable when it is invoked.

        Returns:
            torch.Tensor: Feature embeddings.

        Raises:
            TypeError: If backbone is not None and not callable.
        """

        if self.backbone is None:
            return x

        if not callable(self.backbone):
            raise TypeError(
                "The provided backbone is not callable. Received "
                f"instance of type {type(self.backbone).__name__}."
            )

        return self.backbone(x, **backbone_kwargs if backbone_kwargs else {})
