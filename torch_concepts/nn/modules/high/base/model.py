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
from typing import List, Optional, Mapping, Dict, Union
import functools
import torch
import torch.nn as nn

from .....annotations import Annotations
from .....typing import BackboneType
from .....utils import add_distribution_to_annotations, add_default_properties
from ...utils import with_training_mode
from ...outputs import ModelOutput

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
        Module mapping the raw input (``input_size``) to the ``latent``
        representation (``latent_size``). It runs *inside* the PGM as the
        ``latent | input`` CPD. Can be any ``nn.Module`` (e.g. ResNet, ViT, MLP).
        If None, defaults to ``nn.Identity`` (``latent`` equals the raw input).
        Defaults to None.
    latent_size : int, optional
        Dimensionality of the ``latent`` variable (the backbone's output, and the
        input to the concept encoders). Required when a ``backbone`` is provided
        (cannot be inferred automatically). Defaults to ``input_size`` when no
        backbone is used.

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
    supported_concept_types : frozenset
        Class-level declaration of which concept types this model supports.
        An empty set means no restriction. Concrete models set this to e.g.
        ``frozenset({"binary", "categorical"})``. The recognised strings are the
        keys of :attr:`~torch_concepts.annotations.AxisAnnotation.type_groups`:
        ``"binary"``, ``"categorical"``, ``"continuous"``.
    concept_annotations : AxisAnnotation
        Axis-1 annotations with distribution metadata for each concept.
    concept_names : List[str]
        List of concept variable names from annotations.
    backbone : BackboneType
        Module mapping the raw input to the ``latent`` variable (``nn.Identity`` if
        none was provided). Runs inside the PGM as the ``latent | input`` CPD.
    input_size : int
        Dimensionality of the raw input (the PGM's ``input`` node).
    latent_size : int
        Dimensionality of the ``latent`` variable (input to the concept encoders).

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

    supported_concept_types: frozenset = frozenset()

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
        backbone: Optional[BackboneType] = None,
        latent_size: Optional[int] = None,
        lightning: bool = False,  # Consumed by __new__, included for signature
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._setup_annotations(annotations, variable_distributions)
        self._setup_backbone(backbone, input_size, latent_size)

    def _setup_annotations(
        self,
        annotations: Optional[Annotations],
        variable_distributions: Optional[Mapping],
    ) -> None:
        """Resolve concept annotations and store :attr:`concept_annotations`/:attr:`concept_names`.

        Writes any explicitly passed ``variable_distributions`` into the axis-1
        annotations, fills in default distributions for concepts still missing one,
        and rejects concept types this model cannot handle. Activations are NOT
        stored: they are derived from each variable's distribution when the model
        builds its parametrization.
        """
        if annotations is None:
            return

        annotations = annotations.get_axis_annotation(1)

        # 1. If distributions are explicitly passed, write them into annotations.
        if variable_distributions is not None:
            annotations = add_distribution_to_annotations(annotations, variable_distributions)

        # 2. Fill in default distributions for any concepts still missing one.
        # This also validates that every concept has the metadata we need.
        self.concept_annotations = add_default_properties(annotations)
        self.concept_names = self.concept_annotations.labels

        # 3. Reject annotations that contain concept types this model cannot handle.
        self._validate_concept_types()

    def _setup_backbone(
        self,
        backbone: Optional[BackboneType],
        input_size: int,
        latent_size: Optional[int],
    ) -> None:
        """Store the backbone (raw input → latent) and resolve the sizes.

        The backbone maps whatever the dataloader provides (``input_size``) to the
        ``latent`` representation (``latent_size``); it runs *inside* the PGM as the
        ``latent | input`` CPD. When no backbone is given it defaults to
        ``nn.Identity`` (so :attr:`backbone` is always callable) and ``latent_size``
        falls back to ``input_size``. A custom backbone requires an explicit
        ``latent_size`` since its output dimensionality cannot be inferred.
        """
        self.input_size = input_size
        if backbone is not None:
            if latent_size is None:
                raise ValueError(
                    "Pass `latent_size` when providing a `backbone` — the output "
                    "dimensionality cannot be inferred automatically."
                )
            self._backbone = backbone
            self.latent_size = latent_size
        else:
            self._backbone = nn.Identity()
            self.latent_size = latent_size or input_size

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

    def _validate_concept_types(self) -> None:
        """Raise if the annotations contain concept types this model does not support.

        Does nothing when :attr:`supported_concept_types` is empty (no restriction).
        Uses :attr:`~torch_concepts.annotations.AxisAnnotation.type_groups` to
        determine each concept's type (``"binary"``, ``"categorical"``,
        ``"continuous"``).
        """
        if not self.supported_concept_types:
            return
        groups = self.concept_annotations.type_groups
        unsupported = [
            (type_name, group["labels"])
            for type_name, group in groups.items()
            if group["labels"] and type_name not in self.supported_concept_types
        ]
        if unsupported:
            details = "; ".join(
                f"{t}: {names}" for t, names in unsupported
            )
            raise ValueError(
                f"{type(self).__name__} only supports "
                f"{sorted(self.supported_concept_types)} concept types, but the "
                f"annotations contain: {details}."
            )

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
        
        def _unwrap(fn):
            return fn.func if isinstance(fn, functools.partial) else fn

        if train_inference is not None and _unwrap(train_inference) is not _unwrap(inference):
            raise ValueError(
                f"train_inference ({_unwrap(train_inference).__name__}) must be the same "
                f"class as inference ({_unwrap(inference).__name__}). Different inference "
                "engines for training and evaluation are not yet supported."
            )
        return train_inference if train_inference is not None else inference

    def setup_inference(
        self,
        inference,
        inference_kwargs=None,
        train_inference=None,
        train_inference_kwargs=None,
    ):
        """Instantiate and store the eval/train inference engines.

        Centralises the wiring that every concrete model previously duplicated:
        build ``eval_inference`` from ``inference`` and resolve ``train_inference``
        (falling back to the same class as ``inference``). The concrete ("last")
        child supplies the inference classes; this wraps them around the model's
        ``probabilistic_model``.

        Parameters
        ----------
        inference : type
            Evaluation inference engine class.
        inference_kwargs : dict, optional
            Keyword arguments forwarded to the evaluation engine.
        train_inference : type, optional
            Training inference engine class. Defaults to ``inference``.
        train_inference_kwargs : dict, optional
            Keyword arguments forwarded to the training engine.
        """
        self.eval_inference = inference(
            self.pgm,
            **(inference_kwargs or {}),
        )
        train_inference_cls = self._resolve_train_inference(inference, train_inference)
        self.train_inference = train_inference_cls(
            self.pgm,
            **(train_inference_kwargs or {}),
        )

    def __repr__(self):
        backbone_name = self.backbone.__class__.__name__
        fields = f"backbone={backbone_name}"
        if getattr(self, "input_size", None) is not None:
            fields += f", input_size={self.input_size}"
        if getattr(self, "latent_size", None) is not None:
            fields += f", latent_size={self.latent_size}"
        if getattr(self, "concept_annotations", None) is not None:
            fields += f", n_concepts={len(self.concept_annotations.labels)}"
        if getattr(self, "plate", None) is not None:
            fields += f", plate={self.plate}"
        return f"{self.__class__.__name__}({fields})"

    @property
    def backbone(self) -> BackboneType:
        """The backbone mapping raw input to the latent representation.

        Maps whatever the dataloader provides (``input_size``) to the ``latent``
        variable (``latent_size``); inside the PGM it is the ``latent | input`` CPD.
        Defaults to ``nn.Identity`` when no backbone was provided, so it is always
        callable.

        Returns
        -------
        BackboneType
            Backbone module (e.g., ResNet, ViT, MLP) or ``nn.Identity``.
        """
        return self._backbone

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
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        input: Optional[torch.Tensor] = None,
        **inference_kwargs,
    ) -> ModelOutput:
        """Unified forward pass for all inference engines.

        The active inference engine is selected automatically based on
        ``self.training`` (toggled by ``.train()`` / ``.eval()``). The engine's
        per-variable parameters are assembled into convenient ``probs``/``logits``
        tensors whose columns follow the order of ``query``.

        Parameters
        ----------
        query : List[str]
            Concept names to query.
        evidence : Dict[str, torch.Tensor], optional
            Evidence dict mapping variable names to observed tensors.
        input : torch.Tensor, optional
            Raw input tensor; placed into ``evidence['input']`` so the backbone
            CPD inside the PGM can consume it.

        Returns
        -------
        ModelOutput
            ``params``/``samples`` from the engine.
        """
        if evidence is None:
            evidence = {}
        if input is not None:
            evidence['input'] = input

        result = self.inference.query(
            query=query,
            evidence=evidence,
            **inference_kwargs,
        )

        return ModelOutput(
            params=result.params,
            guide_params=result.guide_params,
            samples=result.samples,
            probabilities=result.probabilities,
            # target=None,  # TODO: to be updated
            # extra=None,  # TODO: to be updated
        )

    def _assemble_predictions(self, query, result, return_logits, return_probs):
        """Concatenate per-variable probabilities (and logits) in query order.

        Each queried variable contributes its ``probs`` parameter (shape
        ``(batch, cardinality)``). Logits are recovered from the probabilities:
        a logit transform for binary variables and a log transform for
        categorical ones (so ``BCEWithLogitsLoss`` / softmax-cross-entropy behave
        as expected).
        """
        if not (return_logits or return_probs):
            return None, None

        eps = 1e-6
        probs_cols, logits_cols = [], []
        axis = getattr(self, "concept_annotations", None)
        for name in query:
            param_dict = result.params[name]
            p = param_dict.get("probs")
            if p is None:
                # Fall back to the first available parameter (e.g. logits).
                p = next(iter(param_dict.values()))
            if return_probs:
                probs_cols.append(p)
            if return_logits:
                cardinality = (
                    int(axis.cardinalities[axis.get_index(name)]) if axis is not None else p.shape[-1]
                )
                clamped = p.clamp(eps, 1.0 - eps)
                logits_cols.append(torch.logit(clamped) if cardinality == 1 else torch.log(clamped))

        probs = torch.cat(probs_cols, dim=-1) if return_probs else None
        logits = torch.cat(logits_cols, dim=-1) if return_logits else None
        return probs, logits

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
