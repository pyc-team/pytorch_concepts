"""Loss functions for concept-based models."""
import inspect
from typing import List, Mapping, Optional
import torch
from torch import nn

from ...nn.modules.utils import GroupConfig
from ...annotations import Annotations, AxisAnnotation
from ...utils import instantiate_from_string
from ...nn.modules.utils import check_collection
from ...nn.modules.mid.constructors.concept_graph import ConceptGraph


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
    (binary, categorical, continuous) using annotation metadata.

    Args:
        annotations (Annotations): Concept annotations with metadata including
            type information for each concept.
        binary (nn.Module, optional): Loss function for binary concepts
            (e.g. ``BCEWithLogitsLoss()``).
        categorical (nn.Module, optional): Loss function for categorical
            concepts (e.g. ``CrossEntropyLoss()``).
        continuous (nn.Module, optional): Loss function for continuous
            concepts (e.g. ``MSELoss()``).  Not yet supported.

    Example:
        >>> from torch_concepts.nn import ConceptLoss
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
        >>> loss_fn = ConceptLoss(
        ...     ann,
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss()
        ... )
        >>>
        >>> predictions = torch.randn(2, 4)
        >>> targets = torch.cat([
        ...     torch.randint(0, 2, (2, 1)),
        ...     torch.randint(0, 3, (2, 1))
        ... ], dim=1)
        >>> loss = loss_fn(predictions, targets)
    """
    def __init__(
        self,
        annotations: Annotations,
        binary: Optional[nn.Module] = None,
        categorical: Optional[nn.Module] = None,
        continuous: Optional[nn.Module] = None,
    ):
        super().__init__()
        fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
        annotations = annotations.get_axis_annotation(axis=1)
        self.fn_collection = check_collection(annotations, fn_collection, 'loss')
        
        # Use cached type_groups from AxisAnnotation
        self.groups = annotations.type_groups
        self.cardinalities = annotations.cardinalities

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
            loss = self.fn_collection.get(t)
            if loss:
                if isinstance(loss, nn.Module):
                    name = loss.__class__.__name__
                elif isinstance(loss, (tuple, list)):
                    name = loss[0].__name__
                else:
                    name = loss.__name__
                parts.append(f"{t}={name}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute total loss across all concept types.
        
        Splits inputs and targets by concept type, computes individual losses,
        and sums them to get the total loss.
        
        Args:
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels/values.
            
        Returns:
            torch.Tensor: Total computed loss (scalar).
        """
        total_loss = 0.0
        
        # Binary concepts
        if self.fn_collection.get('binary'): 
            binary_logits = input[:, self.groups['binary']['logits_idx']]
            binary_targets = target[:, self.groups['binary']['concept_idx']].float()
            total_loss += self.fn_collection['binary'](binary_logits, binary_targets)
        
        # Categorical concepts
        if self.fn_collection.get('categorical'):
            split_tuple = torch.split(
                input[:, self.groups['categorical']['logits_idx']], 
                [self.cardinalities[i] for i in self.groups['categorical']['concept_idx']], 
                dim=1
            )
            padded_logits = [
                nn.functional.pad(
                    logits, 
                    (0, self.max_card - logits.shape[1]), 
                    value=float('-inf')
                ) for logits in split_tuple
            ]
            cat_logits = torch.cat(padded_logits, dim=0)
            cat_targets = target[:, self.groups['categorical']['concept_idx']].T.reshape(-1).long()
            
            total_loss += self.fn_collection['categorical'](cat_logits, cat_targets)
        
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
        binary (nn.Module, optional): Loss function for binary concepts.
        categorical (nn.Module, optional): Loss function for categorical concepts.
        continuous (nn.Module, optional): Loss function for continuous concepts.

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
        binary: Optional[nn.Module] = None,
        categorical: Optional[nn.Module] = None,
        continuous: Optional[nn.Module] = None,
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

        self.concept_loss = ConceptLoss(concept_annotations, binary=binary, categorical=categorical, continuous=continuous)
        self.task_loss = ConceptLoss(task_annotations, binary=binary, categorical=categorical, continuous=continuous)
        self.target_c_idx, self.target_t_idx, self.input_c_idx, self.input_t_idx = get_concept_task_idx(
            annotations, concept_names, task_names
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fn_collection={self.fn_collection})"
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss for concepts and tasks.
        
        Args:
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels/values.
        
        Returns:
            torch.Tensor: Weighted combination of concept and task losses (scalar).
        """
        concept_input = input[:, self.input_c_idx]
        concept_target = target[:, self.target_c_idx]
        task_input = input[:, self.input_t_idx]
        task_target = target[:, self.target_t_idx]
        
        c_loss = self.concept_loss(concept_input, concept_target)
        t_loss = self.task_loss(task_input, task_target)
        
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
        binary (nn.Module, optional): Loss function for binary concepts
            (e.g. ``BCEWithLogitsLoss()``).
        categorical (nn.Module, optional): Loss function for categorical
            concepts (e.g. ``CrossEntropyLoss()``).
        continuous (nn.Module, optional): Loss function for continuous
            concepts (e.g. ``MSELoss()``).  Not yet supported.

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
        binary: Optional[nn.Module] = None,
        categorical: Optional[nn.Module] = None,
        continuous: Optional[nn.Module] = None,
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
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute depth-weighted loss across all concept depths.

        Args:
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels/values.

        Returns:
            torch.Tensor: Total depth-weighted loss (scalar).
        """
        total_loss = torch.tensor(0.0, device=input.device)
        for i, d in enumerate(self._depth_levels):
            sub_input = input[:, self._input_idx[i]]
            sub_target = target[:, self._target_idx[i]]
            sub_loss = getattr(self, f"loss_depth_{d}")
            total_loss = total_loss + self._depth_weights_list[i] * sub_loss(
                sub_input, sub_target
            )
        return total_loss


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


class L1LogitRegularizer(nn.Module):
    """Penalise large logit magnitudes (L1 on predictions)."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.scale * input.abs().mean()
    

class CompositeLoss(nn.Module):
    """Internal modular composition of weighted loss terms.

    Used internally by ``BaseLearner`` when the user passes a list of losses.
    Users should pass ``loss=[term1, term2]`` and ``loss_weights=[w1, w2]`` to
    model constructors rather than instantiating this class directly.

    Sums weighted loss terms, automatically dispatching kwargs to each term
    based on its ``forward()`` signature. Each term's ``forward()`` declares
    the kwargs it needs; ``CompositeLoss`` introspects signatures and passes
    only the matching kwargs. This means ``filter_output_for_loss`` can return
    a superset dict and every term picks what it needs.

    Args:
        terms (List): Loss modules or callables (partials) to sum.
        weights (List[float], optional): Per-term weights.
            Defaults to ``[1.0, ...]``.
        **common_kwargs: Extra kwargs forwarded to any ``terms`` that are
            callables (not yet instantiated). Typically ``annotations``.

    Example::

        # --- User-facing API (via model constructor) ---
        model = ConceptBottleneckModel(
            ...,
            loss=[concept_loss, reg_loss],
            loss_weights=[1.0, 0.5],
        )

        # --- Hydra YAML (via list-based loss config) ---
        # conf/loss/composite.yaml (a YAML list)
        # - _target_: torch_concepts.nn.ConceptLoss
        #   fn_collection: ...
        # - _target_: my_package.MyRegularizer
        #   lambda_: 0.01
        #
        # sweep.yaml:
        #   loss_weights: [1.0, 0.5]
    """

    def __init__(
        self,
        terms: List,
        weights: Optional[List[float]] = None,
        **common_kwargs,
    ):
        super().__init__()

        # Resolve partials / callables with common_kwargs (e.g. annotations)
        resolved: List[nn.Module] = []
        for term in terms:
            if isinstance(term, nn.Module):
                resolved.append(term)
            elif callable(term):
                # functools.partial from Hydra _partial_: true
                # Pass only kwargs the callable accepts
                try:
                    sig = inspect.signature(term)
                    accepted = {
                        k: v for k, v in common_kwargs.items()
                        if k in sig.parameters
                    }
                    resolved.append(term(**accepted))
                except (ValueError, TypeError):
                    # Fallback: pass all kwargs
                    resolved.append(term(**common_kwargs))
            else:
                raise TypeError(
                    f"Each term must be an nn.Module or callable, got {type(term)}"
                )

        # Build unique keys for nn.ModuleDict so each term appears
        # individually in Lightning's model summary.
        name_counts: dict = {}
        term_keys: List[str] = []
        for t in resolved:
            base = t.__class__.__name__
            count = name_counts.get(base, 0)
            name_counts[base] = count + 1
            term_keys.append(f"{base}_{count}" if count > 0 else base)

        self.terms = nn.ModuleDict(dict(zip(term_keys, resolved)))
        self._term_keys = term_keys  # preserve insertion order

        # Weights default to 1.0 per term
        if weights is None:
            weights = [1.0] * len(self.terms)
        if len(weights) != len(self.terms):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of terms ({len(self.terms)})."
            )
        self.register_buffer(
            '_weights',
            torch.tensor(weights, dtype=torch.float32),
        )

        # Introspect each term's forward() for kwarg dispatch
        self._signatures = [
            _get_forward_signature(self.terms[k]) for k in self._term_keys
        ]

    @property
    def weights(self) -> List[float]:
        """Per-term weights as a plain list."""
        return self._weights.tolist()

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts = []
        for key, w in zip(self._term_keys, self.weights):
            parts.append(f"{w}*{key}" if w != 1.0 else key)
        return f"{self.__class__.__name__}({' + '.join(parts)})"

    # ------------------------------------------------------------------
    # kwarg dispatch helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _zero(kwargs):
        """Create a zero scalar on the same device as the first tensor in kwargs."""
        total = torch.tensor(0.0)
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return total.to(v.device)
        return total

    def _dispatch(self, idx, kwargs):
        """Call term *idx* with only the kwargs it accepts."""
        sig, has_var_kw = self._signatures[idx]
        if has_var_kw:
            term_kwargs = kwargs
        else:
            term_kwargs = {k: v for k, v in kwargs.items() if k in sig}
        return self.terms[self._term_keys[idx]](**term_kwargs)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, **kwargs) -> torch.Tensor:
        """Compute weighted sum of all loss terms.

        Each term receives the subset of *kwargs* that its ``forward()``
        signature accepts. Terms whose ``forward`` uses ``**kwargs`` receive
        everything.

        Args:
            **kwargs: Keyword arguments produced by
                ``filter_output_for_loss``.  Typically includes ``input``
                and ``target``, but models may add extra keys (e.g.
                ``embeddings``, ``model``).

        Returns:
            torch.Tensor: Scalar loss (weighted sum of all terms).
        """
        total = self._zero(kwargs)
        for idx in range(len(self._term_keys)):
            total = total + self._weights[idx] * self._dispatch(idx, kwargs)
        return total

    def forward_detailed(self, **kwargs):
        """Compute weighted sum and return per-term losses.

        Same as ``forward`` but additionally returns a dict mapping each
        term's key to its *weighted* loss value, useful for per-term logging.

        Returns:
            Tuple[torch.Tensor, dict[str, torch.Tensor]]:
                (total_loss, {term_key: weighted_loss})
        """
        total = self._zero(kwargs)
        details = {}
        for idx, key in enumerate(self._term_keys):
            val = self._weights[idx] * self._dispatch(idx, kwargs)
            details[key] = val
            total = total + val
        return total, details