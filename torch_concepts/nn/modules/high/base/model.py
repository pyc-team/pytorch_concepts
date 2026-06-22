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
from typing import List, Optional, Mapping, Dict, Type, Union
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....annotations import Annotations
from .....typing import BackboneType
from ...utils import with_training_mode
from ...outputs import ModelOutput, logits_from_params
from ...mid.models.variable import _DEFAULT_DISTRIBUTIONS, _DEFAULT_DIST_KWARGS

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
        Per-instance override of the model's per-*type* distribution policy,
        mapping concept *types* (``'binary'`` / ``'categorical'`` /
        ``'continuous'``) to ``torch.distributions`` classes
        (e.g. ``{'binary': RelaxedBernoulli}``). Merged over the class-level
        :attr:`variable_distributions` default. Distributions are model state —
        they are NOT stored on the annotation. Defaults to None.
    variable_dist_kwargs : Mapping, optional
        Per-instance override of the per-*distribution* keyword arguments,
        mapping a ``torch.distributions`` class to its kwargs
        (e.g. ``{RelaxedBernoulli: {'temperature': 0.7}}`` to change the relaxation
        temperature). Merged over the class-level :attr:`variable_dist_kwargs`
        default. Defaults to None.
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
    - **Concept Distributions**: Distributions are model state, not annotation
      state. Each model has a per-*type* policy (the class attributes
      :attr:`variable_distributions` / :attr:`variable_dist_kwargs`) mapping a
      concept's type (binary/categorical/continuous) to a distribution class;
      a subclass overrides these to change how it models a type, and a caller
      can override per instance with the ``variable_distributions`` argument.
      Activations are derived from the distribution at inference time (not stored).
    - Subclasses must implement ``forward()``.
    - For Lightning training, set lightning=True. The BaseLearner mixin is
      automatically added via ``__new__``.
    - The latent_size attribute is critical for downstream concept encoders
      to determine input dimensionality.

    Examples
    --------
    The model picks a distribution for each concept from its type; override a
    whole type with ``variable_distributions``:

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.distributions import Bernoulli, RelaxedBernoulli
    >>> from torch_concepts.nn import ConceptBottleneckModel
    >>> from torch_concepts.annotations import AxisAnnotation, Annotations
    >>>
    >>> ann = Annotations({
    ...     1: AxisAnnotation(
    ...         labels=['c1', 'c2', 'task'],
    ...         cardinalities=[1, 1, 1],
    ...         types=['binary', 'binary', 'binary'],
    ...     )
    ... })
    >>>
    >>> # Option 1: let the model pick distributions from each concept's type
    >>> model = ConceptBottleneckModel(
    ...     input_size=10, annotations=ann, task_names=['task']
    ... )
    >>> model.variable_distributions['binary'] is Bernoulli
    True
    >>>
    >>> # Option 2: override how this model models a whole type
    >>> model = ConceptBottleneckModel(
    ...     input_size=10, annotations=ann, task_names=['task'],
    ...     variable_distributions={'binary': RelaxedBernoulli},
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

    #: Per-*type* distribution policy: which distribution this model uses for each
    #: concept type (``'binary'`` / ``'categorical'`` / ``'continuous'``).
    #: Distributions are a model concern; a subclass
    #: overrides this to change how it models a type, and a caller can override it
    #: per instance with the ``variable_distributions`` constructor arg.
    variable_distributions: Dict[str, Type] = dict(_DEFAULT_DISTRIBUTIONS)
    #: Default keyword arguments per distribution class (e.g. relaxation temperature).
    variable_dist_kwargs: Dict[Type, dict] = dict(_DEFAULT_DIST_KWARGS)

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
        variable_dist_kwargs: Optional[Mapping] = None,
        backbone: Optional[BackboneType] = None,
        latent_size: Optional[int] = None,
        lightning: bool = False,  # Consumed by __new__, included for signature
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Per-instance overrides of the model's distribution policy: the per-type
        # distribution and the per-distribution kwargs (e.g. relaxation temperature).
        if variable_distributions is not None:
            self.variable_distributions = {**self.variable_distributions, **variable_distributions}
        if variable_dist_kwargs is not None:
            self.variable_dist_kwargs = {**self.variable_dist_kwargs, **variable_dist_kwargs}

        self._setup_annotations(annotations)
        self._setup_backbone(backbone, input_size, latent_size)

    def _setup_annotations(self, annotations: Optional[Annotations]) -> None:
        """Resolve concept annotations and store :attr:`concept_annotations`/:attr:`concept_names`.

        The annotation carries only structural information (labels, types,
        cardinalities); the distribution per concept is a model concern, resolved
        from the per-type policy :attr:`variable_distributions`. Rejects concept
        types this model cannot handle.
        """
        if annotations is None:
            return

        self.concept_annotations = annotations.get_axis_annotation(1)
        self.concept_names = self.concept_annotations.labels

        # Reject annotations that contain concept types this model cannot handle.
        self._validate_concept_types()

    def distribution_of(self, name: str) -> Type:
        """Distribution class this model uses for concept ``name`` (by its type)."""
        return self.variable_distributions[self.concept_annotations.concept(name).type]

    def dist_kwargs_of(self, name: str) -> dict:
        """Distribution keyword arguments this model uses for concept ``name``."""
        return dict(self.variable_dist_kwargs.get(self.distribution_of(name), {}))

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

        If ``train_inference`` is ``None`` it falls back to ``inference``. The
        training and evaluation engines must belong to the *same family*: the two
        classes must be identical or one a subclass of the other (e.g.
        :class:`IndependentInference`, a :class:`DeterministicInference` with
        ``p_int=1``, is a valid training engine for a ``DeterministicInference``
        evaluator). Otherwise a ``ValueError`` is raised.

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
            Resolved training inference class.

        Raises
        ------
        ValueError
            If ``train_inference`` is not in the same class family as ``inference``.
        """

        def _unwrap(fn):
            return fn.func if isinstance(fn, functools.partial) else fn

        if train_inference is not None:
            train_cls, eval_cls = _unwrap(train_inference), _unwrap(inference)
            same_family = (
                train_cls is eval_cls
                or issubclass(train_cls, eval_cls)
                or issubclass(eval_cls, train_cls)
            )
            if not same_family:
                raise ValueError(
                    f"train_inference ({train_cls.__name__}) must be the same class as "
                    f"inference ({eval_cls.__name__}) or a subclass of it. Mixing unrelated "
                    "inference engines for training and evaluation is not yet fully stable hence"
                    "not supported."
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
        ``self.training`` (toggled by ``.train()`` / ``.eval()``). The result is
        returned as raw per-variable parameters under ``ModelOutput.params``:
        ``out.params[name]`` is the queried variable's parameter dict (e.g.
        ``{'logits': ...}`` or ``{'value': ...}``). Callers assemble the columns
        they need, e.g. ``torch.cat([out.params[n]['logits'] for n in query], -1)``.

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
            ``params``/``samples``/``probabilities`` from the engine.
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

        out = ModelOutput(
            params=result.params,
            guide_params=result.guide_params,
            samples=result.samples,
            probabilities=result.probabilities,
        )

        # FIXME: update ModelOutput to generalize beyond logits
        out.logits = logits_from_params(result.params)
        return out

    @functools.cached_property
    def _query_plan(self):
        """Per-concept-variable assembly recipe, computed once (the PGM structure
        is fixed after construction).

        Returns a list ``[(variable_name, [(gt_col_index, cardinality), ...]), ...]``
        over the concept variables only — a plate contributes one entry whose member
        list has all its members, an individual concept contributes a single-member
        entry. :meth:`build_query` applies this without re-deriving structure or
        touching non-concept variables, keeping per-query cost at ``O(n_query)``.
        """
        axis = self.concept_annotations
        return [
            (var.name, [(axis.get_index(m), axis.concept(m).cardinality) for m in var.members])
            for var in self.pgm.variables.values()
            if var.variable_type == "concept"
        ]

    def build_query(self, ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build the full-observation query that fills every concept's tensor.

        Maps the batch concept ground truth (``(batch, n_concepts)`` integer-coded,
        columns in ``concept_annotations.labels`` order) to
        ``{concept_variable_name: tensor}`` for every concept variable in the PGM.
        The query is keyed by the *variable* name (so the inference engine teacher-
        forces it via ``query.get(variable.name)`` — uniformly for plate and
        individual layouts), and each tensor is assembled from the variable's members:
        a categorical member (``cardinality > 1``) is one-hot encoded, a binary /
        scalar member is taken as-is. Evidence (the raw input) is supplied separately.
        """
        query = {}
        for name, members in self._query_plan:
            cols = [
                ground_truth[:, i].float().unsqueeze(-1) if card == 1
                else F.one_hot(ground_truth[:, i].long(), card).float()
                for i, card in members
            ]
            query[name] = torch.cat(cols, dim=-1)
        return query

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
