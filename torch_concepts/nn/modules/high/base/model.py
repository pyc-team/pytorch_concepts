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
from .....tensor import AnnotatedTensor
from .....distributions import Delta
from ...utils import with_training_mode
from ...outputs import ModelOutput
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.sequential import Sequential
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.variable import _DEFAULT_DISTRIBUTIONS, ConceptVariable, EmbeddingVariable

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
    input_size : int or tuple
        Shape of the raw input: an int for flat features, or a tuple (e.g.
        ``(C, H, W)``) when a backbone consumes structured data.
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
    backbone : nn.Module, optional
        Module mapping the raw input (``input_size``) to the ``latent``
        representation (``latent_size``). It runs *inside* the PGM as the
        ``latent | input`` CPD. Can be any ``nn.Module`` (e.g. ResNet, ViT, MLP).
        If None, defaults to ``nn.Identity`` (``latent`` equals the raw input).
        Defaults to None.
    latent_size : int, optional
        Dimensionality of the ``latent`` variable (the backbone's output). 
        Inferred from ``backbone.out_features`` when the backbone exposes it 
        (e.g. :class:`~torch_concepts.Backbone`); otherwise required. 
        Defaults to ``input_size`` when no backbone is used.

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
        keys of :attr:`~torch_concepts.annotations.Annotations.type_groups`:
        ``"binary"``, ``"categorical"``, ``"continuous"``.
    concept_annotations : Annotations
        Concept-axis annotations for each concept.
    concept_names : List[str]
        List of concept variable names from annotations.
    backbone : nn.Module
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
    >>> from torch_concepts.annotations import Annotations
    >>>
    >>> ann = Annotations(
    ...     labels=['c1', 'c2', 'task'],
    ...     cardinalities=[1, 1, 1],
    ...     types=['binary', 'binary', 'binary'],
    ... )
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
    >>> for epoch in range(3):
    ...     optimizer.zero_grad()
    ...     out = model(query=['c1', 'c2', 'task'], input=x)
    ...     loss = loss_fn(out.logits, y)
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
    variable_dist_kwargs: Dict[Type, dict] = dict(DEFAULT_DIST_KWARGS)

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
        backbone: Optional[nn.Module] = None,
        latent_size: Optional[int] = None,
        lightning: bool = False,  # Consumed by __new__, included for signature
        plate: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Per-instance overrides of the model's distribution policy: the per-type
        # distribution and the per-distribution kwargs (e.g. relaxation temperature).
        if variable_distributions is not None:
            self.variable_distributions = {**self.variable_distributions, **variable_distributions}
        if variable_dist_kwargs is not None:
            self.variable_dist_kwargs = {**self.variable_dist_kwargs, **variable_dist_kwargs}

        # Plate preference used by the level factories: None = auto-detect per
        # level, True = force a plate (raise on a heterogeneous level), False =
        # force one variable per concept.
        self._plate_pref = plate

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

        self.concept_annotations = annotations
        self.concept_names = self.concept_annotations.labels

        # Reject annotations that contain concept types this model cannot handle.
        self._validate_concept_types()

    def distribution_of(self, name: str) -> Type:
        """Distribution class this model uses for concept ``name`` (by its type)."""
        return self.variable_distributions[self.concept_annotations.concept(name).type]

    def dist_kwargs_of(self, name: str) -> dict:
        """Distribution keyword arguments this model uses for concept ``name``."""
        return dict(self.variable_dist_kwargs.get(self.distribution_of(name), {}))

    # ------------------------------------------------------------------
    # Level factories (plate grouping) — shared by all concept models
    # ------------------------------------------------------------------
    def _plate_groups(self, names: List[str]) -> list:
        """Partition ``names`` into homogeneous ``(type, cardinality)`` groups.

        Returns ``[((type, cardinality), [names]), ...]`` in first-appearance
        order — one entry per distinct ``(type, cardinality)``. A plate must be
        homogeneous, so this is the **minimum** number of plates that can cover the
        level: fully homogeneous → 1 group, two homogeneous families (e.g. 10
        Bernoulli + 10 identical categorical) → 2 groups, all distinct → N groups.
        Reads only annotation scalars — it builds no ``Variable`` objects — so it
        stays cheap even for very large levels.
        """
        axis = self.concept_annotations
        groups: Dict[tuple, List[str]] = {}
        for n in names:
            c = axis.concept(n)
            groups.setdefault((c.type, c.cardinality), []).append(n)
        return list(groups.items())

    def _plate_layout(self, names: List[str], plate_name: str) -> list:
        """Resolve how a level is laid out, shared by the variable factories.

        Returns a list of ``(kind, name, members)`` where ``kind`` is ``"plate"``
        or ``"individual"``. Honours the ``plate`` preference (:attr:`_plate_pref`):

        * ``None`` (default) / ``True`` — group homogeneous concepts into the
          minimum number of plates; even a lone concept becomes a single-member
          plate, so the level is always a list of plates and ``plate_name`` is
          always used.
        * ``False`` — no plates: one individual variable per concept, named after
          the concept.

        A single plate keeps the bare ``plate_name``; multiple plates are suffixed
        with their ``type`` and ``cardinality`` (e.g. ``concepts_binary_1``) so the
        names are unique.
        """
        if self._plate_pref is False:
            return [("individual", n, [n]) for n in names]
        # None / True: always plates (a lone concept is a single-member plate).
        groups = self._plate_groups(names)
        single = len(groups) == 1
        return [
            ("plate", plate_name if single else f"{plate_name}_{ctype}_{card}", members)
            for (ctype, card), members in groups
        ]

    def _make_concept_plate(self, members: List[str], name: str) -> ConceptVariable:
        """A single plate :class:`ConceptVariable` over homogeneous ``members``."""
        c0 = self.concept_annotations.concept(members[0])
        return ConceptVariable(
            names=name,
            members=list(members),
            distribution=self.distribution_of(c0.name),
            dist_kwargs=self.dist_kwargs_of(c0.name),
            size=c0.cardinality,
        )

    def _make_concept_variable(self, name: str) -> ConceptVariable:
        """A single (non-plate) :class:`ConceptVariable` for concept ``name``."""
        c = self.concept_annotations.concept(name)
        return ConceptVariable(
            names=c.name,
            distribution=self.distribution_of(c.name),
            dist_kwargs=self.dist_kwargs_of(c.name),
            size=c.cardinality,
        )

    def build_concept_variables(self, names: List[str], plate_name: str) -> List[ConceptVariable]:
        """Build the concept variable(s) for a set of concepts, as a list of ``ConceptVariable``.

        Uses :meth:`_plate_layout`. By default (``plate`` ``None``/``True``)
        homogeneous concepts are grouped into the minimum number of plates — each a
        :class:`ConceptVariable` with one member per grouped concept, and even a
        lone concept a single-member plate. With ``plate=False`` each concept is an
        individual variable named after itself. Returning a list lets callers wire
        the CPDs and the Bayesian network identically however the set splits.
        """
        out: List[ConceptVariable] = []
        for kind, name, members in self._plate_layout(names, plate_name):
            if kind == "plate":
                out.append(self._make_concept_plate(members, name))
            else:
                out.append(self._make_concept_variable(name))
        return out

    @staticmethod
    def _embedding_rows(concept) -> int:
        """State-embedding rows a concept contributes: 2 for binary, else ``cardinality``.

        A binary concept scores one Bernoulli probability but is mixed from two
        independent contexts :math:`w^+`, :math:`w^-` (Espinosa Zarlenga et al.,
        NeurIPS 2022). Must stay in lock-step with
        :class:`~torch_concepts.nn.modules.low.predictors.mix.MixConceptEmbedding`,
        which expands a binary concept's group to the same 2 rows.
        """
        return 2 if concept.is_binary else concept.cardinality

    @staticmethod
    def build_concept_head(cvar: ConceptVariable, evar: EmbeddingVariable) -> nn.Module:
        """Decode ``cvar``'s scores from ``evar``'s state embeddings.

        ``(B, k*rows, m) -> (B, k*out)`` for ``k = len(cvar.members)``, ``rows``
        embedding rows per member (:meth:`_embedding_rows`) and ``out =
        cvar.member_size``. One shared ``Linear(m, 1)`` per state row when
        ``rows == out`` (categorical, continuous — the CEM head); for a binary
        concept, whose 2 rows feed 1 score, the CEM paper's joint
        :math:`\\Psi_i([w^+, w^-])` instead — a ``Linear(2m, 1)`` reading both.

        Dropping the ``rows == out`` guard would make *every* head the joint
        ``Linear(card*m, card)`` of Ismail et al., ICLR 2024.
        """
        rows, m = evar.shape[0] // len(cvar.members), evar.shape[1]
        out = cvar.member_size
        if rows == out:
            return Sequential(
                LinearEmbeddingToConcept(in_embeddings=m, out_concepts=1),
                nn.Flatten(start_dim=-2),  # (B, k*rows, 1) -> (B, k*rows)
            )
        # Note: this branch starts with nn.Unflatten, so ParametricCPD's
        # `_module_input_names` gives it STANDARD aggregation (mod(t)) where the
        # branch above gets PyC aggregation (mod(embeddings=t)). Same lone parent
        # tensor either way — only the calling convention differs.
        return Sequential(
            nn.Unflatten(-2, (-1, rows)),   # (B, k*rows, m) -> (B, k, rows, m)
            nn.Flatten(start_dim=-2),       # -> (B, k, rows*m)
            LinearEmbeddingToConcept(in_embeddings=rows * m, out_concepts=out),
            nn.Flatten(start_dim=-2),       # (B, k, out) -> (B, k*out)
        )

    def build_concept_embedding_variables(
        self,
        names: List[str],
        embedding_size: int,
        plate_name: str,
        name_fmt: str = "{}_embedding",
    ) -> List[EmbeddingVariable]:
        """Build the per-concept state-embedding variable(s), aligned with the concepts.

        Each concept contributes one state embedding per state — except a
        binary concept, which contributes **two** (:math:`w^+`, :math:`w^-`;
        see :meth:`_embedding_rows`) even though it is a single Bernoulli.
        This uses the **same** :meth:`_plate_layout` as
        :meth:`build_concept_variables`, so — called with the same ``names`` — the
        returned list aligns element-by-element with the concept variables: group
        ``i``'s embeddings feed group ``i``'s concepts. Per group it returns:

        * a plate of ``k`` homogeneous concepts → one :class:`EmbeddingVariable` of
          shape ``(k * rows, embedding_size)`` (the members' state embeddings
          stacked into one matrix, ``rows`` from :meth:`_embedding_rows`); a lone
          concept is a single-member plate of shape ``(rows, embedding_size)``;
        * with ``plate=False``, one ``EmbeddingVariable`` per concept of shape
          ``(rows, embedding_size)``, named via ``name_fmt``.

        Only *concept* embeddings belong here. Global/shared embeddings (``input``,
        ``latent``, model-wide latents) are not per-concept and carry no grouping —
        build those directly with :class:`EmbeddingVariable`. ``plate_name`` must
        differ from the concepts' ``plate_name`` so the plate nodes do not collide.
        """
        out: List[EmbeddingVariable] = []
        for kind, name, members in self._plate_layout(names, plate_name):
            if kind == "plate":
                c0 = self.concept_annotations.concept(members[0])
                shape = (len(members) * self._embedding_rows(c0), embedding_size)
            else:
                name = name_fmt.format(name)
                c = self.concept_annotations.concept(members[0])
                shape = (self._embedding_rows(c), embedding_size)
            out.append(EmbeddingVariable(name, distribution=Delta, shape=shape))
        return out

    def _setup_backbone(
        self,
        backbone: Optional[nn.Module],
        input_size: int,
        latent_size: Optional[int],
    ) -> None:
        """Store the backbone (raw input → latent) and resolve the sizes.

        The backbone maps whatever the dataloader provides (``input_size``) to the
        ``latent`` representation (``latent_size``); it runs *inside* the PGM as the
        ``latent | input`` CPD. When no backbone is given it defaults to
        ``nn.Identity`` (so :attr:`backbone` is always callable) and ``latent_size``
        falls back to ``input_size``. With a custom backbone, ``latent_size``
        is taken from the backbone's ``out_features`` when it exposes one
        (e.g. :class:`~torch_concepts.Backbone`, ``nn.Linear``); otherwise it
        must be passed explicitly.
        """
        self.input_size = input_size
        if backbone is not None:
            self._backbone = backbone
            self.latent_size = latent_size or getattr(backbone, 'out_features', None)
            if self.latent_size is None:
                raise ValueError(
                    "Pass `latent_size` when the `backbone` does not expose "
                    "`out_features`."
                )
        else:
            self._backbone = nn.Identity()
            self.latent_size = latent_size or input_size
            if not isinstance(self.latent_size, int):
                raise ValueError(
                    "Without a `backbone` the latent equals the raw input, so "
                    f"`input_size` must be an int; got {self.latent_size!r}. Add a "
                    "`backbone` to map multi-dimensional inputs to a vector latent."
                )

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
        Uses :attr:`pyc.Annotations.type_groups` to
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

    def setup_inference(
        self,
        inference=None,
        inference_kwargs=None,
        train_inference=None,
        train_inference_kwargs=None,
    ):
        """Set up — or later swap — the eval/train inference engines.

        Called once at construction to wire the engines around the model's
        ``pgm``, and callable again on an already-instantiated model to change
        inference. Only the engines whose class is passed are (re)built; any
        engine left as ``None`` keeps its current instance, so you can replace
        just the eval or just the train engine. On the very first setup the
        training engine falls back to the ``inference`` class (so passing only
        ``inference`` wires both). Because the :attr:`inference` property reads
        ``eval_inference``/``train_inference``, a replacement is used on the
        next :meth:`forward`.

        Parameters
        ----------
        inference : type, optional
            Evaluation inference engine class. Engine left unchanged if None.
        inference_kwargs : dict, optional
            Keyword arguments forwarded to the evaluation engine.
        train_inference : type, optional
            Training inference engine class. On first setup defaults to
            ``inference``; on later swaps the engine is left unchanged if None.
        train_inference_kwargs : dict, optional
            Keyword arguments forwarded to the training engine.
        """
        if inference is not None:
            self.eval_inference = inference(self.pgm, **(inference_kwargs or {}))
        # First setup: train falls back to the eval class. Later swaps: replace
        # the train engine only when a class is passed explicitly.
        first_setup = not hasattr(self, "train_inference")
        train_inference_cls = train_inference or (inference if first_setup else None)
        if train_inference_cls is not None:
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
    def backbone(self) -> nn.Module:
        """The backbone mapping raw input to the latent representation.

        Maps whatever the dataloader provides (``input_size``) to the ``latent``
        variable (``latent_size``); inside the PGM it is the ``latent | input`` CPD.
        Defaults to ``nn.Identity`` when no backbone was provided, so it is always
        callable.

        Returns
        -------
        nn.Module
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
        keyed by *quantity*: ``out.logits`` (equivalently ``out.params['logits']``)
        is one annotated tensor spanning every queried concept, sliceable by name
        — ``out.logits['c1']`` — with the columns of a variable that reports a
        different quantity living under its own key (``out.loc`` / ``out.scale``).

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

        return ModelOutput(
            params=result.params,
            guide_params=result.guide_params,
            samples=result.samples,
            probabilities=result.probabilities,
        )

    @functools.cached_property
    def _query_plan(self):
        """Per-concept-variable assembly recipe, computed once (the PGM structure
        is fixed after construction).

        Returns a list ``[(variable_name, [(gt_col_index, cardinality), ...]), ...]``
        over the concept variables only — a plate contributes one entry whose member
        list has all its members, an individual concept contributes a single-member
        entry. :meth:`fully_observed_query` applies this without re-deriving structure or
        touching non-concept variables, keeping per-query cost at ``O(n_query)``.
        """
        axis = self.concept_annotations
        return [
            (var.name, [(axis.get_index(m), axis.concept(m).cardinality) for m in var.members])
            for var in self.pgm.variables.values()
            if var.variable_type == "concept"
        ]

    @functools.cached_property
    def _query_segments(self):
        """Precompute the gather/one-hot plan used by :meth:`fully_observed_query`.

        Run-length-encodes each concept variable's members (from :attr:`_query_plan`)
        into segments, in member order: a maximal run of consecutive cardinality-1
        members (binary and/or continuous, which need no expansion) is merged into
        one ``'plain'`` segment fetched with a single vectorized gather; each
        cardinality>1 member (categorical) becomes its own ``'onehot'`` segment.
        Computed once and cached, since the PGM structure is fixed after construction.

        Returns:
            dict[str, list[tuple]]: Map from concept variable name to its ordered
            list of segments. Each segment is one of:

            - ``('plain', index_tensor)`` — a :class:`torch.LongTensor` of
              ground-truth column indices to gather directly (no expansion).
            - ``('onehot', (index, cardinality))`` — a single ground-truth column
              index and its number of classes, to be one-hot encoded.
        """
        segments = {}
        for name, members in self._query_plan:
            var_segments = []
            run = []
            for i, card in members:
                if card == 1:
                    run.append(i)
                    continue
                if run:
                    var_segments.append(('plain', torch.tensor(run, dtype=torch.long)))
                    run = []
                var_segments.append(('onehot', (i, card)))
            if run:
                var_segments.append(('plain', torch.tensor(run, dtype=torch.long)))
            segments[name] = var_segments
        return segments

    def fully_observed_query(self, ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build the teacher-forcing query from full concept ground truth.

        Assembles, for every concept variable in the PGM, the tensor the inference
        engine should force it to during a fully-observed pass (e.g. teacher forcing
        in training/eval). Each variable's tensor is built from :attr:`_query_segments`:
        binary/continuous columns are gathered directly, categorical columns are
        one-hot encoded. Evidence (the raw input) is supplied separately by the caller.

        Args:
            ground_truth: Batch concept ground truth of shape ``(batch, n_concepts)``,
                integer-coded, with columns ordered as in ``concept_annotations.labels``.
                May be a plain :class:`torch.Tensor` or an :class:`AnnotatedTensor`
                (unwrapped internally).

        Returns:
            dict[str, torch.Tensor]: Map from concept variable name to its query
            tensor, keyed for lookup via ``query.get(variable.name)``.
        """
        raw = ground_truth.tensor if isinstance(ground_truth, AnnotatedTensor) else ground_truth
        query = {}
        for name, segments in self._query_segments.items():
            if len(segments) == 1 and segments[0][0] == 'plain':
                query[name] = raw[:, segments[0][1]].float()
                continue
            pieces = []
            for kind, payload in segments:
                if kind == 'plain':
                    pieces.append(raw[:, payload].float())
                else:
                    i, card = payload
                    pieces.append(F.one_hot(raw[:, i].long(), card).float())
            query[name] = torch.cat(pieces, dim=-1)
        return query

    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        """Prepare ground-truth labels for loss/metrics.

        Returns the target as a concept-space :class:`AnnotatedTensor` (one column
        per concept), so losses and metrics can align it to the predictions by
        name. A target that already carries an annotation is returned unchanged.
        Override in subclasses that predict a subset of concepts (e.g. task-only).

        Parameters
        ----------
        target : torch.Tensor
            Raw ground-truth labels from the batch.

        Returns
        -------
        AnnotatedTensor or None
            Concept-space annotated target.
        """
        if target is None or hasattr(target, 'annotation'):
            return target
        return AnnotatedTensor(target, self.concept_annotations.to_concept_space(), axis=-1)
