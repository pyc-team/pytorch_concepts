"""Loss functions for concept-based models."""
import inspect
from typing import List, Mapping, Optional, Union
import torch
from torch import nn

from ...nn.modules.utils import GroupConfig
from ...annotations import Annotations, AxisAnnotation
from ...utils import instantiate_from_string
from ...nn.modules.utils import check_collection
from ...nn.modules.mid.constructors.concept_graph import ConceptGraph


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


def get_concept_task_idx(annotations: AxisAnnotation, concepts: List[str], tasks: List[str]):
    """Get concept and task indices at both concept-level and logit-level."""
    # Concept-level indices
    concepts_idxs = [annotations.get_index(name) for name in concepts]
    tasks_idxs = [annotations.get_index(name) for name in tasks]
    
    # Logit-level indices using cached get_slice
    concepts_logits = annotations.get_slice(concepts)
    tasks_logits = annotations.get_slice(tasks)
    
    return concepts_idxs, tasks_idxs, concepts_logits, tasks_logits

class ConceptLoss(nn.Module):
    """
    Concept loss for concept-based models.

    Automatically routes to appropriate loss functions based on concept types
    (binary, categorical, continuous) using annotation metadata. Each type
    accepts either a single loss module or a list of loss modules with
    optional per-term weights, enabling type-specific composition (e.g.
    adding a regularizer only to binary concepts).

    Args:
        annotations (Annotations): Concept annotations with metadata including
            type information for each concept.
        binary (nn.Module or list of nn.Module, optional): Loss function(s)
            for binary concepts. A single module (e.g. ``BCEWithLogitsLoss()``)
            or a list of modules to be summed.
        categorical (nn.Module or list of nn.Module, optional): Loss function(s)
            for categorical concepts. A single module (e.g.
            ``CrossEntropyLoss()``) or a list of modules.
        continuous (nn.Module or list of nn.Module, optional): Loss function(s)
            for continuous concepts (e.g. ``MSELoss()``).  Not yet supported.
        binary_weights (list of float, optional): Per-term weights when
            ``binary`` is a list. Defaults to ``[1.0, ...]``.
        categorical_weights (list of float, optional): Per-term weights when
            ``categorical`` is a list. Defaults to ``[1.0, ...]``.
        continuous_weights (list of float, optional): Per-term weights when
            ``continuous`` is a list. Defaults to ``[1.0, ...]``.

    Example:
        >>> from torch_concepts.nn import ConceptLoss, L1LogitRegularizer
        >>> from torch_concepts import Annotations, AxisAnnotation
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> from torch.distributions import Bernoulli, Categorical
        >>>
        >>> ann = Annotations({1: AxisAnnotation(
        ...     labels=['is_round', 'color'],
        ...     cardinalities=[1, 3],
        ...     metadata={
        ...         'is_round': {'type': 'discrete', 'distribution': Bernoulli},
        ...         'color': {'type': 'discrete', 'distribution': Categorical}
        ...     }
        ... )})
        >>>
        >>> # Single loss per type (backward compatible)
        >>> loss_fn = ConceptLoss(
        ...     ann,
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss()
        ... )
        >>>
        >>> # Composite loss per type with weights
        >>> loss_fn = ConceptLoss(
        ...     ann,
        ...     binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        ...     binary_weights=[1.0, 0.5],
        ...     categorical=CrossEntropyLoss()
        ... )
    """
    def __init__(
        self,
        annotations: Annotations,
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        # Normalize to lists
        binary, binary_weights = _normalize_loss_terms(binary, binary_weights)
        categorical, categorical_weights = _normalize_loss_terms(categorical, categorical_weights)
        continuous, continuous_weights = _normalize_loss_terms(continuous, continuous_weights)
        
        # Validate against annotations (check_collection checks None vs not-None)
        fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
        annotations = annotations.get_axis_annotation(axis=1)
        self.fn_collection = check_collection(annotations, fn_collection, 'loss')
        
        # Use cached type_groups from AxisAnnotation
        self.groups = annotations.type_groups
        self.cardinalities = annotations.cardinalities

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
                self._type_signatures[type_name] = [
                    _get_forward_signature(m) for m in terms
                ]

        # For categorical loss, precompute max cardinality for padding
        if self.fn_collection.get('categorical'):
            cat_idx = self.groups['categorical']['concept_idx']
            self.max_card = max([self.cardinalities[i] for i in cat_idx])

        if self.fn_collection.get('continuous'):
            cont_idx = self.groups['continuous']['concept_idx']
            self.max_dim = max([self.cardinalities[i] for i in cont_idx])

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
        """
        terms = getattr(self, f'_{type_name}_terms')
        weights = self._type_weights[type_name]
        signatures = self._type_signatures[type_name]
        
        total = torch.tensor(0.0, device=kwargs['input'].device)
        
        for module, weight, (sig, has_var_kw) in zip(terms, weights, signatures):
            if has_var_kw:
                term_kwargs = dict(kwargs)
            else:
                term_kwargs = {k: v for k, v in kwargs.items() if k in sig}
            total = total + weight * module(**term_kwargs)
        
        return total

    def _prepare_categorical(self, input: torch.Tensor, target: torch.Tensor):
        """Pad and stack categorical logits/targets for summary computation.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (padded_logits, targets) ready
            for loss functions like ``CrossEntropyLoss``.
        """
        cat_concept_idx = self.groups['categorical']['concept_idx']
        split_tuple = torch.split(
            input[:, self.groups['categorical']['logits_idx']],
            [self.cardinalities[i] for i in cat_concept_idx],
            dim=1,
        )
        padded_logits = [
            nn.functional.pad(
                logits,
                (0, self.max_card - logits.shape[1]),
                value=float('-inf'),
            )
            for logits in split_tuple
        ]
        cat_logits = torch.cat(padded_logits, dim=0)
        cat_targets = target[:, cat_concept_idx].T.reshape(-1).long()
        return cat_logits, cat_targets

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute total loss across all concept types.
        
        Splits ``input`` and ``target`` by concept type, merges them with any
        extra kwargs, computes individual losses (each a weighted sum of its
        terms dispatched by signature), and sums them.
        
        Args:
            **kwargs: Must include ``input`` (logits) and ``target`` (labels).
                Any additional kwargs (e.g. ``embeddings``, ``model``) are
                forwarded to loss terms whose ``forward()`` signature accepts
                them.
            
        Returns:
            torch.Tensor: Total computed loss (scalar).
        """
        input = kwargs['input']
        target = kwargs['target']
        extra = {k: v for k, v in kwargs.items() if k not in ('input', 'target')}
        
        total_loss = torch.tensor(0.0, device=input.device)
        
        # Binary concepts
        if self.fn_collection.get('binary'): 
            binary_logits = input[:, self.groups['binary']['logits_idx']]
            binary_targets = target[:, self.groups['binary']['concept_idx']].float()
            total_loss = total_loss + self._compute_type_loss('binary', {
                'input': binary_logits, 'target': binary_targets, **extra
            })
        
        # Categorical concepts
        if self.fn_collection.get('categorical'):
            cat_logits, cat_targets = self._prepare_categorical(input, target)
            total_loss = total_loss + self._compute_type_loss('categorical', {
                'input': cat_logits, 'target': cat_targets, **extra
            })
        
        # Continuous concepts
        if self.fn_collection.get('continuous'):
            raise NotImplementedError("Continuous concepts not yet implemented.")
        
        return total_loss


class WeightedConceptLoss(nn.Module):
    """
    Weighted concept loss for concept-based models.

    Computes a weighted combination of concept and task losses.

    Args:
        annotations (Annotations): Annotations object with concept metadata.
        concept_weight (float): Weight for concept loss.
        task_weight (float): Weight for task loss.
        task_names (List[str]): List of task concept names.
        binary (nn.Module or list of nn.Module, optional): Loss function(s) for binary concepts.
        categorical (nn.Module or list of nn.Module, optional): Loss function(s) for categorical concepts.
        continuous (nn.Module or list of nn.Module, optional): Loss function(s) for continuous concepts.
        binary_weights (list of float, optional): Per-term weights when ``binary`` is a list.
        categorical_weights (list of float, optional): Per-term weights when ``categorical`` is a list.
        continuous_weights (list of float, optional): Per-term weights when ``continuous`` is a list.

    Example:
        >>> from torch_concepts.nn.modules.loss import WeightedConceptLoss
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> from torch_concepts.annotations import AxisAnnotation, Annotations
        >>> ann = Annotations({1: AxisAnnotation(labels=['c1', 'c2', 'task'], cardinalities=[1, 3, 1])})
        >>> loss_fn = WeightedConceptLoss(
        ...     ann, concept_weight=0.7, task_weight=0.3,
        ...     task_names=['task'], binary=BCEWithLogitsLoss()
        ... )
        >>> input = torch.randn(2, 5)
        >>> target = torch.randint(0, 2, (2, 3))
        >>> loss = loss_fn(input, target)
    """
    def __init__(
        self,
        annotations: Annotations,
        concept_weight: float,
        task_weight: float,
        task_names: List[str],
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.concept_weight = concept_weight
        self.task_weight = task_weight
        fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
        self.fn_collection = fn_collection
        annotations = annotations.get_axis_annotation(axis=1)
        concept_names = [name for name in annotations.labels if name not in task_names]
        task_annotations = Annotations({1:annotations.subset(task_names)})
        concept_annotations = Annotations({1:annotations.subset(concept_names)})

        self.concept_loss = ConceptLoss(
            concept_annotations, binary=binary, categorical=categorical, continuous=continuous,
            binary_weights=binary_weights, categorical_weights=categorical_weights,
            continuous_weights=continuous_weights,
        )
        self.task_loss = ConceptLoss(
            task_annotations, binary=binary, categorical=categorical, continuous=continuous,
            binary_weights=binary_weights, categorical_weights=categorical_weights,
            continuous_weights=continuous_weights,
        )
        self.target_c_idx, self.target_t_idx, self.input_c_idx, self.input_t_idx = get_concept_task_idx(
            annotations, concept_names, task_names
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fn_collection={self.fn_collection})"
    
    def forward(self, **kwargs) -> torch.Tensor:
        """Compute weighted loss for concepts and tasks.
        
        Args:
            **kwargs: Must include ``input`` (logits) and ``target`` (labels).
                Extra kwargs are forwarded to the inner ConceptLoss instances.
        
        Returns:
            torch.Tensor: Weighted combination of concept and task losses (scalar).
        """
        input = kwargs['input']
        target = kwargs['target']
        extra = {k: v for k, v in kwargs.items() if k not in ('input', 'target')}
        
        concept_input = input[:, self.input_c_idx]
        concept_target = target[:, self.target_c_idx]
        task_input = input[:, self.input_t_idx]
        task_target = target[:, self.target_t_idx]
        
        c_loss = self.concept_loss(input=concept_input, target=concept_target, **extra)
        t_loss = self.task_loss(input=task_input, target=task_target, **extra)
        
        return c_loss * self.concept_weight + t_loss * self.task_weight


class DepthWeightedConceptLoss(nn.Module):
    """Depth-weighted concept loss for graph-structured concept models.

    Applies different weights to concept losses based on their depth
    in a directed acyclic graph (DAG).  Concepts at the graph sources
    (roots, depth 0) receive ``source_weight``; at each subsequent depth
    level the weight is multiplied by ``depth_decay``.

    Weight at depth *d* = ``source_weight * depth_decay ** d``

    Args:
        annotations (Annotations): Concept annotations with metadata.
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
            for continuous concepts (e.g. ``MSELoss()``).  Not yet supported.
        binary_weights (list of float, optional): Per-term weights when
            ``binary`` is a list.
        categorical_weights (list of float, optional): Per-term weights when
            ``categorical`` is a list.
        continuous_weights (list of float, optional): Per-term weights when
            ``continuous`` is a list.

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.loss import DepthWeightedConceptLoss
        >>> from torch_concepts.annotations import Annotations, AxisAnnotation
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import ConceptGraph
        >>> from torch.distributions import Bernoulli
        >>>
        >>> ann = Annotations({1: AxisAnnotation(
        ...     labels=['A', 'B', 'C'],
        ...     cardinalities=[1, 1, 1],
        ...     metadata={
        ...         'A': {'type': 'discrete', 'distribution': Bernoulli},
        ...         'B': {'type': 'discrete', 'distribution': Bernoulli},
        ...         'C': {'type': 'discrete', 'distribution': Bernoulli},
        ...     }
        ... )})
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        >>> loss_fn = DepthWeightedConceptLoss(
        ...     ann, graph,
        ...     source_weight=1.0, depth_decay=0.5,
        ...     binary=torch.nn.BCEWithLogitsLoss()
        ... )
        >>> preds = torch.randn(4, 3)
        >>> targets = torch.randint(0, 2, (4, 3)).float()
        >>> loss = loss_fn(preds, targets)
    """

    def __init__(
        self,
        annotations: Annotations,
        graph: ConceptGraph,
        source_weight: float = 1.0,
        depth_decay: float = 0.5,
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.source_weight = source_weight
        self.depth_decay = depth_decay

        axis = annotations.get_axis_annotation(axis=1)
        concept_names = list(axis.labels)
        concept_set = set(concept_names)
        # Compute levels from graph
        depth_levels = graph.get_levels()

        # For each depth level store a ConceptLoss sub-module,
        # concept-level indices (target slicing), logit-level indices
        # (input slicing), and the corresponding weight.
        self._depth_levels: List[int] = []
        self._depth_weights_list: List[float] = []
        self._target_idx: List[List[int]] = []
        self._input_idx: List[List[int]] = []

        for d, level_names in enumerate(depth_levels):
            # Keep only concepts that appear in the annotations
            names = [n for n in level_names if n in concept_set]
            if not names:
                continue
            sub_ann = Annotations({1: axis.subset(names)})

            key = f"loss_depth_{d}"
            sub_loss = ConceptLoss(
                sub_ann,
                binary=binary,
                categorical=categorical,
                continuous=continuous,
                binary_weights=binary_weights,
                categorical_weights=categorical_weights,
                continuous_weights=continuous_weights,
            )
            setattr(self, key, sub_loss)

            self._depth_levels.append(d)
            self._target_idx.append([axis.get_index(n) for n in names])
            self._input_idx.append(axis.get_slice(names))
            self._depth_weights_list.append(source_weight * (depth_decay ** d))

        # Concepts not in the graph get depth 0
        graph_names = {n for level in depth_levels for n in level}
        missing = [n for n in concept_names if n not in graph_names]
        if missing:
            sub_ann = Annotations({1: axis.subset(missing)})
            key = "loss_depth_0"
            if not hasattr(self, key):
                sub_loss = ConceptLoss(
                    sub_ann,
                    binary=binary,
                    categorical=categorical,
                    continuous=continuous,
                    binary_weights=binary_weights,
                    categorical_weights=categorical_weights,
                    continuous_weights=continuous_weights,
                )
                setattr(self, key, sub_loss)
                self._depth_levels.insert(0, 0)
                self._target_idx.insert(0, [axis.get_index(n) for n in missing])
                self._input_idx.insert(0, axis.get_slice(missing))
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
    def forward(self, **kwargs) -> torch.Tensor:
        """Compute depth-weighted loss across all concept depths.

        Args:
            **kwargs: Must include ``input`` (logits) and ``target`` (labels).
                Extra kwargs are forwarded to the inner ConceptLoss instances.

        Returns:
            torch.Tensor: Total depth-weighted loss (scalar).
        """
        input = kwargs['input']
        target = kwargs['target']
        extra = {k: v for k, v in kwargs.items() if k not in ('input', 'target')}
        
        total_loss = torch.tensor(0.0, device=input.device)
        for i, d in enumerate(self._depth_levels):
            sub_input = input[:, self._input_idx[i]]
            sub_target = target[:, self._target_idx[i]]
            sub_loss = getattr(self, f"loss_depth_{d}")
            total_loss = total_loss + self._depth_weights_list[i] * sub_loss(
                input=sub_input, target=sub_target, **extra
            )
        return total_loss


class L1LogitRegularizer(nn.Module):
    """Penalise large logit magnitudes (L1 on predictions)."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(input)
        if mask.any():
            return self.scale * input[mask].abs().mean()
        return torch.tensor(0.0, device=input.device)