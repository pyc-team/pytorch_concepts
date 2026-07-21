"""Loss functions for concept-based models."""
import inspect
import warnings
from typing import List, Mapping, Optional, Union
import torch
from torch import nn

from .utils import GroupConfig
from .outputs import ModelOutput
from ...utils import instantiate_from_string
from ...concept_graph import ConceptGraph


def _get_forward_signature(module: nn.Module):
    """Introspect forward() to get accepted parameter names and whether it has **kwargs.
    
    Returns:
        Tuple[set, bool]: (set of parameter names, has_var_keyword)
    """
    params = inspect.signature(module.forward).parameters
    names = set()
    has_var_keyword = False
    for name, param in params.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            names.add(name)
    return names, has_var_keyword


def _normalize_loss_terms(terms, weights):
    """Normalize loss terms and weights to consistent list form.
    
    Args:
        terms: A single nn.Module, a list of nn.Module, or None.
        weights: A list of floats, or None.
        
    Returns:
        Tuple of (list_of_modules, list_of_weights), or (None, None) if terms is None.
    """
    if terms is None:
        return None, None
    if isinstance(terms, nn.Module):
        terms = [terms]
    if not isinstance(terms, (list, tuple)):
        raise TypeError(
            f"Loss terms must be an nn.Module or a list of nn.Module, got {type(terms)}"
        )
    if weights is None:
        weights = [1.0] * len(terms)
    if len(weights) != len(terms):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of loss terms ({len(terms)})."
        )
    return list(terms), list(weights)


def subset_output(output: ModelOutput, names: List[str]) -> ModelOutput:
    """A :class:`ModelOutput` restricted to the concepts in ``names``.

    Every quantity tensor and the target are sliced by concept name via their
    annotations, so the result carries whatever quantities those concepts report
    (``logits``/``probs`` and/or ``loc``/``scale``). Used by the composite losses
    to route a shared output to their sub-losses.
    """
    sub = ModelOutput(extra=output.extra)
    if output.target is not None:
        present = [n for n in names if n in output.target.annotation.label_to_index]
        sub.target = output.target[present]
    for quantity, tensor in output.params.items():
        present = [n for n in names if n in tensor.annotation.label_to_index]
        if present:
            sub.params[quantity] = tensor[present]
    return sub


class TypeAwareLoss(nn.Module):
    """Base for losses that route by concept type, consuming a whole
    :class:`ModelOutput` in ``forward`` (optionally with an explicit target)
    rather than a plain ``loss(input, target)``. The learner uses this to route
    the model output to the loss.
    """


class ConceptLoss(TypeAwareLoss):
    """
    Concept loss for concept-based models.

    Routes to the appropriate loss function based on the concept type
    (binary, categorical, continuous) read from the annotated model output.
    Each type accepts either a single loss module or a list of loss modules
    with optional per-term weights, enabling type-specific composition (e.g.
    adding a regularizer only to binary concepts).

    Args:
        binary (nn.Module or list of nn.Module, optional): Loss function(s)
            for binary concepts. A single module (e.g. ``BCEWithLogitsLoss()``)
            or a list of modules to be summed.
        categorical (nn.Module or list of nn.Module, optional): Loss function(s)
            for categorical concepts. A single module (e.g.
            ``CrossEntropyLoss()``) or a list of modules to be summed.
        continuous (nn.Module or list of nn.Module, optional): Loss function(s)
            for continuous concepts, scored on ``loc``. A single module 
            (e.g. ``MSELoss()``) or a list of modules to be summed.
        binary_weights (list of float, optional): Per-term weights when
            ``binary`` is a list. Defaults to ``[1.0, ...]``.
        categorical_weights (list of float, optional): Per-term weights when
            ``categorical`` is a list. Defaults to ``[1.0, ...]``.
        continuous_weights (list of float, optional): Per-term weights when
            ``continuous`` is a list. Defaults to ``[1.0, ...]``.
        binary_param (str): Output quantity to read binary predictions from.
            Default ``'logits'``.
        categorical_param (str): Output quantity to read categorical predictions
            from. Default ``'logits'``.
        continuous_param (str): Output quantity to read continuous predictions
            from. Default ``'loc'``.

    Example:
        >>> from torch_concepts.nn import ConceptLoss, L1LogitRegularizer
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>>
        >>> # Single loss per type
        >>> loss_fn = ConceptLoss(
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss()
        ... )
        >>>
        >>> # Composite loss per type with weights
        >>> loss_fn = ConceptLoss(
        ...     binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        ...     binary_weights=[1.0, 0.5],
        ...     categorical=CrossEntropyLoss()
        ... )
    """
    def __init__(
        self,
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
        binary_param: Optional[str] = 'logits',
        categorical_param: Optional[str] = 'logits',
        continuous_param: Optional[str] = 'loc',
    ):
        super().__init__()

        binary, binary_weights = _normalize_loss_terms(binary, binary_weights)
        categorical, categorical_weights = _normalize_loss_terms(categorical, categorical_weights)
        continuous, continuous_weights = _normalize_loss_terms(continuous, continuous_weights)
        self.fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)

        # Register modules, weights, and signatures per type
        self._type_weights = {}
        self._type_signatures = {}
        weights_map = {
            'binary': binary_weights,
            'categorical': categorical_weights,
            'continuous': continuous_weights,
        }
        for type_name in ['binary', 'categorical', 'continuous']:
            terms = self.fn_collection.get(type_name)
            if terms is not None:
                # Register as nn.ModuleList for proper parameter tracking
                setattr(self, f'_{type_name}_terms', nn.ModuleList(terms))
                self._type_weights[type_name] = weights_map[type_name]
                # fill each loss type with (set of parameter names, has_var_keyword)
                self._type_signatures[type_name] = [
                    _get_forward_signature(m) for m in terms
                ]
        
        self.binary_param = binary_param
        self.categorical_param = categorical_param
        self.continuous_param = continuous_param

        # Static categorical padding layout, keyed by the tuple of per-concept
        # cardinalities (see _prepare_categorical). Data-independent, so rebuilt
        # only when a new cardinality signature is seen.
        self._cat_pad_cache = {}

    def __repr__(self) -> str:
        types = ['binary', 'categorical', 'continuous']
        parts = []
        for t in types:
            terms = self.fn_collection.get(t)
            if terms is not None:
                weights = self._type_weights[t]
                if len(terms) == 1 and weights[0] == 1.0:
                    name = terms[0].__class__.__name__
                    parts.append(f"{t}={name}")
                else:
                    term_strs = []
                    for m, w in zip(terms, weights):
                        n = m.__class__.__name__
                        term_strs.append(f"{w}*{n}" if w != 1.0 else n)
                    parts.append(f"{t}=[{' + '.join(term_strs)}]")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _compute_type_loss(self, type_name: str, kwargs: dict) -> torch.Tensor:
        """Compute weighted sum of loss terms for a specific concept type.
        
        Each term receives only the kwargs its ``forward()`` signature accepts.
        If ``padding_mask`` is present in *kwargs* but a term's signature does
        not accept it (and has no ``**kwargs``), a warning is emitted so that
        users are aware their custom loss/regularizer is receiving padded
        values without explicit masking information.
        """
        terms = getattr(self, f'_{type_name}_terms')
        weights = self._type_weights[type_name]
        signatures = self._type_signatures[type_name]
        
        has_padding = 'padding_mask' in kwargs
        total = torch.tensor(0.0, device=kwargs['input'].device)
        
        for module, weight, (sig, has_var_kw) in zip(terms, weights, signatures):
            if has_var_kw:
                term_kwargs = dict(kwargs)
            else:
                term_kwargs = {k: v for k, v in kwargs.items() if k in sig}
                if has_padding and 'padding_mask' not in sig and 'target' not in sig:
                    warnings.warn(
                        f"{module.__class__.__name__} does not accept a "
                        f"'padding_mask' parameter. Categorical concept "
                        f"logits are padded with -inf for concepts with "
                        f"cardinality < max_cardinality. If this module "
                        f"could be affected by this, add a 'padding_mask' parameter "
                        f"to its forward() to handle padded positions "
                        f"correctly.",
                        stacklevel=2,
                    )
            total = total + weight * module(**term_kwargs)
        
        return total

    def _prepare_categorical(self, cat_logits: torch.Tensor, cat_target: torch.Tensor):
        """Pad and stack categorical logits/targets for CrossEntropy-style terms.

        ``cat_logits`` (logit-space) and ``cat_target`` (concept-space) are the
        categorical slices from :meth:`AnnotatedTensor.split_by_type`, already in
        the same concept order; per-concept widths come from ``cat_logits``'s
        annotation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(padded_logits, targets, padding_mask)`` ready for loss
                functions like ``CrossEntropyLoss``.  ``padding_mask`` is a
                boolean tensor of the same shape as ``padded_logits`` that is
                ``True`` for real logit positions and ``False`` for padding.
        """
        cards = list(cat_logits.annotation.cardinalities)

        # The padding layout (max width and which columns are real per concept)
        # depends only on the cardinalities, so cache it per cardinality signature.
        key = tuple(cards)
        template = self._cat_pad_cache.get(key)
        if template is None:
            max_card = max(cards)
            # col_valid[i, j] is True where column j is a real class of concept i.
            col_valid = torch.zeros(len(cards), max_card, dtype=torch.bool)
            for i, c in enumerate(cards):
                col_valid[i, :c] = True
            template = (max_card, col_valid)
            self._cat_pad_cache[key] = template
        max_card, col_valid = template

        batch = cat_logits.shape[0]
        padded_logits = [
            nn.functional.pad(logits, (0, max_card - logits.shape[1]), value=float('-inf'))
            for logits in torch.split(cat_logits, cards, dim=1)
        ]
        cat_logits_out = torch.cat(padded_logits, dim=0)
        # Repeat each concept's column-validity row over the batch to match the
        # concept-major row order of ``cat_logits_out``.
        cat_mask = col_valid.to(cat_logits.device).repeat_interleave(batch, dim=0)
        cat_targets = cat_target.T.reshape(-1).long()
        return cat_logits_out, cat_targets, cat_mask

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        """Compute total loss across all concept types.

        Each type reads its predictions from a configurable quantity
        (``binary_param`` / ``categorical_param`` / ``continuous_param``) and is
        aligned to the target by concept name.

        Args:
            output (ModelOutput): Structured model output containing
                ``logits``, ``target``, and optionally ``extras``.
            target (AnnotatedTensor, optional): Concept-space ground truth.
                Defaults to ``output.target``.

        Returns:
            torch.Tensor: Total computed loss (scalar).
        """
        extra = dict(output.extra) if output.extra else {}
        target = target if target is not None else output.target

        # Each type reads its predictions from a configurable quantity: binary and
        # categorical are sliced by type out of their (discrete) quantity; continuous
        # is taken directly from its quantity. split_by_type() is memoised per
        # quantity so a shared discrete quantity (the default, logits for both) is
        # split only once.
        splits = {}
        def split_for(param):
            if param not in splits:
                q = output.params.get(param)
                splits[param] = q.split_by_type() if q is not None else {}
            return splits[param]

        binary = split_for(self.binary_param).get('binary')
        categorical = split_for(self.categorical_param).get('categorical')
        continuous = output.params.get(self.continuous_param)

        contributions = []

        if self.fn_collection.get('binary') and binary is not None:
            contributions.append(self._compute_type_loss('binary', {
                'input': binary,
                'target': target[binary.annotation.labels].float(),
                **extra
            }))

        if self.fn_collection.get('categorical') and categorical is not None:
            cat_logits, cat_targets, cat_mask = self._prepare_categorical(
                categorical, target[categorical.annotation.labels]
            )
            contributions.append(self._compute_type_loss('categorical', {
                'input': cat_logits,
                'target': cat_targets,
                'padding_mask': cat_mask,
                **extra
            }))

        if self.fn_collection.get('continuous') and continuous is not None:
            kwargs = {'input': continuous, 'target': target[continuous.annotation.labels], **extra}
            if output.scale is not None:
                kwargs['scale'] = output.scale
            contributions.append(self._compute_type_loss('continuous', kwargs))

        return sum(contributions) if contributions else torch.zeros(())


class WeightedConceptLoss(TypeAwareLoss):
    """
    Weighted concept loss for concept-based models.

    Computes a weighted combination of concept and task losses.

    Args:
        concept_weight (float): Weight for concept loss.
        task_weight (float): Weight for task loss.
        task_names (List[str]): List of task concept names.
        binary (nn.Module or list of nn.Module, optional): Loss function(s) for binary concepts.
        categorical (nn.Module or list of nn.Module, optional): Loss function(s) for categorical concepts.
        continuous (nn.Module or list of nn.Module, optional): Loss function(s) for continuous concepts.
        binary_weights (list of float, optional): Per-term weights when ``binary`` is a list.
        categorical_weights (list of float, optional): Per-term weights when ``categorical`` is a list.
        continuous_weights (list of float, optional): Per-term weights when ``continuous`` is a list.
        binary_param (str): Output quantity for binary predictions. Default ``'logits'``.
        categorical_param (str): Output quantity for categorical predictions. Default ``'logits'``.
        continuous_param (str): Output quantity for continuous predictions. Default ``'loc'``.

    Example:
        >>> from torch_concepts.nn.modules.loss import WeightedConceptLoss
        >>> from torch.nn import BCEWithLogitsLoss
        >>> loss_fn = WeightedConceptLoss(
        ...     concept_weight=0.7, task_weight=0.3,
        ...     task_names=['task'], binary=BCEWithLogitsLoss()
        ... )
        >>> loss = loss_fn(model_output)  # doctest: +SKIP
    """
    def __init__(
        self,
        concept_weight: float,
        task_weight: float,
        task_names: List[str],
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
        binary_param: str = 'logits',
        categorical_param: str = 'logits',
        continuous_param: str = 'loc',
    ):
        super().__init__()
        self.concept_weight = concept_weight
        self.task_weight = task_weight
        self.fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
        # Concepts are every output name that is not a task; both are sliced from
        # the output by name at forward.
        self.task_names = list(task_names)

        loss_kwargs = dict(
            binary=binary, categorical=categorical, continuous=continuous,
            binary_weights=binary_weights, categorical_weights=categorical_weights,
            continuous_weights=continuous_weights,
            binary_param=binary_param, categorical_param=categorical_param,
            continuous_param=continuous_param,
        )
        self.concept_loss = ConceptLoss(**loss_kwargs)
        self.task_loss = ConceptLoss(**loss_kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fn_collection={self.fn_collection})"
    
    def forward(self, output: ModelOutput) -> torch.Tensor:
        """Compute weighted loss for concepts and tasks.

        Args:
            output (ModelOutput): Structured model output containing
                ``logits``, ``target``, and optionally ``extras``.
        
        Returns:
            torch.Tensor: Weighted combination of concept and task losses (scalar).
        """
        concept_names = [n for n in output.target.annotation.labels if n not in self.task_names]
        c_sub = subset_output(output, concept_names)
        t_sub = subset_output(output, self.task_names)
        c_loss = self.concept_loss(c_sub)
        t_loss = self.task_loss(t_sub)

        return c_loss * self.concept_weight + t_loss * self.task_weight


class DepthWeightedConceptLoss(TypeAwareLoss):
    """Depth-weighted concept loss for graph-structured concept models.

    Applies different weights to concept losses based on their depth
    in a directed acyclic graph (DAG).  Concepts at the graph sources
    (roots, depth 0) receive ``source_weight``; at each subsequent depth
    level the weight is multiplied by ``depth_decay``.

    Weight at depth *d* = ``source_weight * depth_decay ** d``

    Args:
        graph (ConceptGraph): DAG defining structure among concepts.
        source_weight (float): Weight applied to loss terms at depth 0
            (graph sources).  Default ``1.0``.
        depth_decay (float): Multiplicative factor applied at every
            additional depth level.  Values < 1 down-weight deeper
            concepts; values > 1 up-weight them.  Default ``0.5``.
        binary (nn.Module or list of nn.Module, optional): Loss function(s)
            for binary concepts (e.g. ``BCEWithLogitsLoss()``).
        categorical (nn.Module or list of nn.Module, optional): Loss function(s)
            for categorical concepts (e.g. ``CrossEntropyLoss()``).
        continuous (nn.Module or list of nn.Module, optional): Loss function(s)
            for continuous concepts (e.g. ``MSELoss()``), scored on ``loc``.
        binary_weights (list of float, optional): Per-term weights when
            ``binary`` is a list.
        categorical_weights (list of float, optional): Per-term weights when
            ``categorical`` is a list.
        continuous_weights (list of float, optional): Per-term weights when
            ``continuous`` is a list.
        binary_param (str): Output quantity for binary predictions. Default ``'logits'``.
        categorical_param (str): Output quantity for categorical predictions. Default ``'logits'``.
        continuous_param (str): Output quantity for continuous predictions. Default ``'loc'``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.loss import DepthWeightedConceptLoss
        >>> from torch_concepts import ConceptGraph
        >>>
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        >>> loss_fn = DepthWeightedConceptLoss(
        ...     graph,
        ...     source_weight=1.0, depth_decay=0.5,
        ...     binary=torch.nn.BCEWithLogitsLoss()
        ... )
        >>> loss = loss_fn(model_output)  # doctest: +SKIP
    """

    def __init__(
        self,
        graph: ConceptGraph,
        source_weight: float = 1.0,
        depth_decay: float = 0.5,
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
        binary_param: str = 'logits',
        categorical_param: str = 'logits',
        continuous_param: str = 'loc',
    ):
        super().__init__()
        self.source_weight = source_weight
        self.depth_decay = depth_decay

        depth_levels = graph.get_levels()
        self._graph_names = {n for level in depth_levels for n in level}

        # Per depth level: a ConceptLoss sub-module, the concept names at that
        # level (used to slice the model output by name at forward), and a weight.
        self._depth_levels: List[int] = []
        self._depth_weights_list: List[float] = []
        self._level_names: List[List[str]] = []

        loss_kwargs = dict(
            binary=binary, categorical=categorical, continuous=continuous,
            binary_weights=binary_weights, categorical_weights=categorical_weights,
            continuous_weights=continuous_weights,
            binary_param=binary_param, categorical_param=categorical_param,
            continuous_param=continuous_param,
        )

        def _make_sub_loss():
            return ConceptLoss(**loss_kwargs)

        for d, level_names in enumerate(depth_levels):
            if not level_names:
                continue
            setattr(self, f"loss_depth_{d}", _make_sub_loss())
            self._depth_levels.append(d)
            self._level_names.append(list(level_names))
            self._depth_weights_list.append(source_weight * (depth_decay ** d))

        # Depth 0 also absorbs concepts absent from the graph (resolved at forward).
        if not hasattr(self, "loss_depth_0"):
            setattr(self, "loss_depth_0", _make_sub_loss())
            self._depth_levels.insert(0, 0)
            self._level_names.insert(0, [])
            self._depth_weights_list.insert(0, source_weight)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts = []
        for d, w in zip(self._depth_levels, self._depth_weights_list):
            parts.append(f"depth_{d}: weight={w:.4g}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, output: ModelOutput) -> torch.Tensor:
        """Compute depth-weighted loss across all concept depths.

        Args:
            output (ModelOutput): Structured model output containing
                ``logits``, ``target``, and optionally ``extras``.

        Returns:
            torch.Tensor: Total depth-weighted loss (scalar).
        """
        any_pred = output.logits if output.logits is not None else output.loc
        total_loss = torch.tensor(0.0, device=any_pred.device)
        # Concepts in the output but absent from the graph are scored at depth 0.
        missing = [n for n in output.target.annotation.labels if n not in self._graph_names]
        for i, d in enumerate(self._depth_levels):
            names = self._level_names[i] + missing if d == 0 else self._level_names[i]
            sub = subset_output(output, names)
            if not sub.params:
                continue
            sub_loss = getattr(self, f"loss_depth_{d}")
            total_loss = total_loss + self._depth_weights_list[i] * sub_loss(sub)
        return total_loss


class L1LogitRegularizer(nn.Module):
    """Penalise large logit magnitudes via L1 regularisation.

    Computes ``scale * mean(|input|)`` over all valid (non-padded)
    positions.  When used as a categorical loss term inside
    :class:`ConceptLoss`, a ``padding_mask`` is automatically provided
    to distinguish real logits from padding.

    :class:`ConceptLoss`::

        loss_fn = ConceptLoss(
            binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
            binary_weights=[1.0, 0.5],
        )

    Args:
        scale (float): Multiplicative factor applied to the L1 mean.
            Default ``1.0``.

    Returns:
        torch.Tensor: Scalar regularisation loss.
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            mask = padding_mask
        else:
            mask = torch.isfinite(input)
        if mask.any():
            return self.scale * input[mask].abs().mean()
        return torch.tensor(0.0, device=input.device)