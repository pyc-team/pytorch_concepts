
from contextlib import nullcontext
from typing import Callable, Iterable, Optional, Union, List, Tuple
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from copy import deepcopy

from ...annotations import Annotations
from .low.base.intervention import InterventionPolicy, InterventionStrategy, intervention
from .low.intervention.policy.uniform import UniformPolicy
from .low.intervention.strategy.do import DoIntervention
from .outputs import (CONTINUOUS_QUANTITIES, DISCRETE_QUANTITIES, InferenceOutput,
                      resolve_quantity, supervised_subset)
from .utils import by_type, check_collection


def clone_metric(metric):
    """Clone and reset a metric for independent tracking across splits."""
    metric = metric.clone()
    metric.reset()
    return metric


class ConceptMetrics(nn.Module):
    """Type-aware metric manager for concept-based models.

    Routes predictions to the correct metrics based on concept type
    (binary / categorical / continuous), read at update time from the
    annotated model output. The concept structure (names, types,
    cardinalities) comes from ``annotations`` at construction, so every
    ``torchmetrics.Metric`` submodule exists from the start. Supports
    summary metrics (aggregated per type) and per-concept metrics, with
    independent state for each data split (train / val / test).

    Args:
        annotations (Annotations): Concept annotations (axis 1) with labels,
            cardinalities, and types (``'binary'``, ``'categorical'``, or ``'continuous'``).
        binary: Metric specs for binary concepts (cardinality 1).
        categorical: Metric specs for categorical concepts (cardinality > 1).
        continuous: Metric specs for continuous concepts, scored on ``loc``.
        summary (bool): Compute summary metrics aggregated across all
            concepts of each type.  Default ``True``.
        per_concept (bool | list[str]): ``False`` (default) disables
            per-concept tracking; ``True`` tracks every concept; a list
            of names tracks only those concepts.

    Each metric spec is a dict ``{name: spec}`` where ``spec`` is:

    * A pre-instantiated ``torchmetrics.Metric``.
    * A ``(MetricClass, kwargs)`` tuple — ``num_classes`` is injected
      automatically for categorical concepts.
    * A non-instantiated ``MetricClass``.

    Example::

        metrics = ConceptMetrics(
            annotations=annotations,
            binary={"accuracy": BinaryAccuracy()},
            categorical={"accuracy": (MulticlassAccuracy, {"average": "micro"})},
        )
        metrics.update(model_output)
        results = metrics.compute()   # {"SUMMARY-binary_accuracy": ..., ...}
        metrics.reset()
    """

    def __init__(
        self,
        annotations: Annotations,
        binary: Union[nn.Module, Tuple[nn.Module, dict]] = None,
        categorical: Union[nn.Module, Tuple[nn.Module, dict]] = None,
        continuous: Union[nn.Module, Tuple[nn.Module, dict]] = None,
        summary: bool = True,
        per_concept: Union[bool, List[str]] = False,
        prefix: Optional[str] = None
    ):
        super().__init__()

        self.summary = summary
        self.per_concept = per_concept

        self.concept_annotations = annotations
        self.concept_names = list(annotations.labels)
        self.n_concepts = len(self.concept_names)
        self.cardinalities = annotations.cardinalities
        self.types = list(annotations.types)

        # Use cached type_groups from Annotations
        self.groups = annotations.type_groups

        # Validate and filter metrics configuration
        fn_collection = by_type(binary, categorical, continuous)
        self.fn_collection = check_collection(annotations, fn_collection, 'metrics')

        # Pre-compute max cardinality for categorical concepts
        self.max_card = None
        if self.fn_collection.get('categorical'):
            self.max_card = max([self.cardinalities[i]
                                for i in self.groups['categorical']['concept_idx']])

        # Determine which concepts to track for per-concept metrics
        if self.per_concept:
            if isinstance(self.per_concept, bool):
                self._concepts_to_trace = list(self.concept_names)
            elif isinstance(self.per_concept, list):
                invalid = [n for n in self.per_concept if n not in self.concept_names]
                if invalid:
                    raise ValueError(
                        f"Concept names not found in annotations: {invalid}"
                    )
                self._concepts_to_trace = self.per_concept
            else:
                raise ValueError(
                    "per_concept must be either a bool or a list of concept names."
                )
        else:
            self._concepts_to_trace = []

        # Setup separate MetricCollections per type and per concept
        pfx = f"{prefix}/" if prefix else ""
        self._prefix = pfx
        summary_b, summary_c, summary_cont, per_concept_dict = self._setup_metrics()

        # Summary collections: one MetricCollection per concept type
        self.binary = MetricCollection(
            metrics=summary_b, prefix=f"{pfx}SUMMARY-binary_"
        ) if summary_b else MetricCollection({})

        self.categorical = MetricCollection(
            metrics=summary_c, prefix=f"{pfx}SUMMARY-categorical_"
        ) if summary_c else MetricCollection({})

        self.continuous = MetricCollection(
            metrics=summary_cont, prefix=f"{pfx}SUMMARY-continuous_"
        ) if summary_cont else MetricCollection({})

        # Per-concept collections: one MetricCollection per tracked concept
        self._per_concept = nn.ModuleDict({
            name: MetricCollection(metrics=metrics, prefix=f"{pfx}{name}_")
            for name, metrics in per_concept_dict.items()
        })

    def __repr__(self) -> str:
        metric_info = {
            k: [
                (m.__class__.__name__ if isinstance(m, Metric)
                 else m[0].__name__ if isinstance(m, (tuple, list))
                 else m.__name__)
                for m in v.values()
            ]
            for k, v in self.fn_collection.items() if v
        }
        metrics_str = ', '.join(f"{k}=[{','.join(v)}]" for k, v in metric_info.items())
        return (f"{self.__class__.__name__}(n_concepts={self.n_concepts}, "
                f"metrics={{{metrics_str}}}, summary={self.summary}, "
                f"per_concept={self.per_concept})")

    @property
    def collection(self):
        """Return all non-empty sub-collections as a dict."""
        result = {}
        if len(self.binary):
            result['binary'] = self.binary
        if len(self.categorical):
            result['categorical'] = self.categorical
        if len(self.continuous):
            result['continuous'] = self.continuous
        for name, coll in self._per_concept.items():
            if len(coll):
                result[name] = coll
        return result

    def clone(self, prefix=None):
        """Create an independent copy with fresh state and optional new prefix.

        Args:
            prefix: New prefix for metric keys. If None, keeps the original.
        """
        cloned = deepcopy(self)
        if prefix is not None:
            pfx = f"{prefix}/" if prefix else ""
            cloned._prefix = pfx
            if len(cloned.binary):
                cloned.binary.prefix = f"{pfx}SUMMARY-binary_"
            if len(cloned.categorical):
                cloned.categorical.prefix = f"{pfx}SUMMARY-categorical_"
            if len(cloned.continuous):
                cloned.continuous.prefix = f"{pfx}SUMMARY-continuous_"
            for name, coll in cloned._per_concept.items():
                coll.prefix = f"{pfx}{name}_"
        cloned.reset()
        return cloned

    def _instantiate_metric(self, metric_spec, concept_specific_kwargs=None):
        """Instantiate a metric from either an instance or a class+kwargs tuple/list.

        Args:
            metric_spec: Either a Metric instance, a tuple/list (MetricClass, kwargs_dict),
                or a MetricClass (will be instantiated with concept_specific_kwargs only).
            concept_specific_kwargs (dict): Concept-specific parameters to merge with user kwargs.

        Returns:
            Metric: Instantiated metric.

        Raises:
            ValueError: If user provides 'num_classes' in kwargs (it's set automatically).
        """
        if isinstance(metric_spec, Metric):
            return metric_spec.clone()
        elif isinstance(metric_spec, (tuple, list)) and len(metric_spec) == 2:
            # (MetricClass, user_kwargs)
            metric_class, user_kwargs = metric_spec

            # Check if user provided num_classes when it will be set automatically
            if 'num_classes' in user_kwargs and concept_specific_kwargs and 'num_classes' in concept_specific_kwargs:
                raise ValueError(
                    f"'num_classes' should not be provided in metric kwargs. "
                    f"ConceptMetrics automatically sets 'num_classes' based on concept cardinality."
                )

            merged_kwargs = {**(concept_specific_kwargs or {}), **user_kwargs}
            return metric_class(**merged_kwargs)
        else:
            # Just a class, use concept_specific_kwargs only
            return metric_spec(**(concept_specific_kwargs or {}))

    def _setup_metrics(self):
        """Instantiate metrics, separated into summary and per-concept groups.

        Returns:
            Tuple of (summary_binary, summary_categorical, summary_continuous,
            per_concept) where per_concept maps concept name to metric dict.
        """
        summary_binary = {}
        summary_categorical = {}
        summary_continuous = {}
        per_concept = {}

        # Summary metrics (keyed by metric name; prefix added by MetricCollection)
        if self.summary:
            if self.fn_collection.get('binary'):
                for name, spec in self.fn_collection['binary'].items():
                    summary_binary[name] = self._instantiate_metric(spec)

            if self.fn_collection.get('categorical'):
                for name, spec in self.fn_collection['categorical'].items():
                    summary_categorical[name] = self._instantiate_metric(
                        spec, concept_specific_kwargs={'num_classes': self.max_card}
                    )

            if self.fn_collection.get('continuous'):
                for name, spec in self.fn_collection['continuous'].items():
                    summary_continuous[name] = self._instantiate_metric(spec)

        # Per-concept metrics (one dict per concept)
        for concept_name in self._concepts_to_trace:
            c_idx = self.concept_names.index(concept_name)
            c_type = self.types[c_idx]
            card = self.cardinalities[c_idx]

            concept_metrics = {}
            if c_type == 'binary':
                for name, spec in self.fn_collection.get('binary', {}).items():
                    concept_metrics[name] = self._instantiate_metric(spec)
            elif c_type == 'categorical':
                for name, spec in self.fn_collection.get('categorical', {}).items():
                    concept_metrics[name] = self._instantiate_metric(
                        spec, concept_specific_kwargs={'num_classes': card}
                    )
            elif c_type == 'continuous':
                for name, spec in self.fn_collection.get('continuous', {}).items():
                    concept_metrics[name] = self._instantiate_metric(spec)

            if concept_metrics:
                per_concept[concept_name] = concept_metrics

        return summary_binary, summary_categorical, summary_continuous, per_concept

    @staticmethod
    def _concept_prediction(out, name, concept_type):
        """Prediction columns for one concept, taken from the quantity its type
        reports: ``loc`` or ``value`` for continuous, otherwise the discrete param (logits/probs).

        Addressed quantity-first (``params[quantity][name]``) rather than
        variable-first: a concept may legitimately be *named* like a distribution
        parameter — dSprites has one called ``scale`` — and ``params[name]`` would
        then resolve to the quantity, not to that concept's columns.
        """
        quantities = (
            CONTINUOUS_QUANTITIES if concept_type == 'continuous' else ('logits', 'probs')
        )
        for quantity in quantities:
            tensor = out.params.get(quantity)
            if tensor is not None and name in tensor.annotations.label_to_index:
                # Hand torchmetrics a plain tensor; the annotation is no longer
                # needed and would make each internal torch.* op pay the
                # __torch_function__ cost. Discrete ones as probabilities (see update).
                pred = tensor[name].tensor
                if quantity == 'logits':
                    pred = pred.sigmoid() if concept_type == 'binary' else pred.softmax(-1)
                return pred
        raise KeyError(
            f"ConceptMetrics: no {' or '.join(quantities)} reported for {concept_type} "
            f"concept {name!r}; the output carries {tuple(out.params)}."
        )

    def _prepare_categorical(self, cat_logits, cat_target):
        """Pad and stack categorical logits/targets for the summary metric.

        ``cat_logits`` (logit-space) and ``cat_target`` (concept-space) are the
        categorical slices from :meth:`AnnotatedTensor.categorical`, already in
        the same concept order; per-concept widths come from ``cat_logits``'s own
        annotation.
        """
        # Unwrap to plain tensors up front: past this point the annotation has
        # served its purpose (concept ordering), and every torch.* op below on an
        # AnnotatedTensor would otherwise re-enter its __torch_function__ hook.
        cards = list(cat_logits.annotations.cardinalities)
        # Unwrap to plain tensors and fold any leading (batch-like) dimensions
        # into one batch axis, so the layout below is always (batch, width).
        # Both are flattened the same way, so their rows stay aligned.
        cat_logits = getattr(cat_logits, "tensor", cat_logits).reshape(-1, cat_logits.shape[-1])
        cat_target = getattr(cat_target, "tensor", cat_target).reshape(-1, cat_target.shape[-1])
        split_tuple = torch.split(cat_logits, cards, dim=1)
        padded_logits = [
            nn.functional.pad(
                logits,
                (0, self.max_card - logits.shape[1]),
                value=float('-inf')
            ) for logits in split_tuple
        ]
        cat_pred = torch.cat(padded_logits, dim=0)
        cat_target = cat_target.T.reshape(-1).long()
        return cat_pred, cat_target

    def update(self, preds, target: torch.Tensor):
        """Update metrics by routing predictions to the correct type collection.

        Summary metrics receive aggregated data for all concepts of a type.
        Per-concept metrics receive individual concept data.

        Args:
            preds: A ``InferenceOutput`` or a single :class:`AnnotatedTensor` of
                discrete predictions.
            target: Annotated concept-space ground truth.
        """
        # A InferenceOutput carries one AnnotatedTensor per quantity (logits/probs for
        # discrete concepts, loc/scale for continuous); each type is scored on the
        # quantity that represents it. A bare tensor is taken as the discrete params.
        if isinstance(preds, InferenceOutput):
            out = preds
        else:
            out = InferenceOutput()
            out.logits = preds

        # The same quantity ConceptLoss reads (see resolve_quantity): `loc` for a
        # Normal, `value` for a Delta; logits before probs.
        discrete = resolve_quantity(out.params, None, DISCRETE_QUANTITIES, target)
        continuous = resolve_quantity(out.params, None, CONTINUOUS_QUANTITIES, target)
        # torchmetrics guesses logits vs probabilities from the value range, so
        # logits that all fall in [0, 1] would read as probabilities: hand it
        # probabilities always (sigmoid / softmax below).
        from_logits = discrete is out.logits
        any_pred = discrete if discrete is not None else continuous
        if any_pred is None or any_pred.shape[0] == 0:
            return
        # Variables the target provides no truth for (a generative model's
        # reconstructed observation, say) are dropped rather than looked up.
        binary = supervised_subset(
            discrete.binary() if discrete is not None else None, target)
        categorical = supervised_subset(
            discrete.categorical() if discrete is not None else None, target)
        continuous = supervised_subset(continuous, target)

        # Summary metrics — one MetricCollection.update() per type, each scored on
        # the quantity it reports and aligned to the target by concept name.
        if self.summary:
            # Predictions/targets are unwrapped to plain tensors before every
            # torchmetrics update: the annotation is only used to align preds to
            # the target by concept name, and passing an AnnotatedTensor into
            # torchmetrics makes each internal torch.* op re-enter the
            # __torch_function__ unwrap hook (the dominant per-update cost).
            if binary is not None and len(self.binary):
                pred = binary.tensor.sigmoid() if from_logits else binary.tensor
                self.binary.update(pred, target[binary.annotations].tensor.float())
            if categorical is not None and len(self.categorical):
                cat_pred, cat_target = self._prepare_categorical(
                    categorical, target[categorical.annotations])
                # The -inf padding becomes probability 0 either way.
                cat_pred = cat_pred.softmax(-1) if from_logits else cat_pred.clamp_min(0)
                self.categorical.update(cat_pred, cat_target)
            if continuous is not None and len(self.continuous):
                self.continuous.update(
                    continuous.tensor, target[continuous.annotations].tensor)

        # Per-concept metrics — read each concept from its type's quantity.
        for concept_name, collection in self._per_concept.items():
            c_type = self.types[self.concept_names.index(concept_name)]
            c_pred = self._concept_prediction(out, concept_name, c_type)
            c_target = target[concept_name].tensor
            if c_type == 'binary':
                collection.update(c_pred, c_target.float())
            elif c_type == 'categorical':
                # torchmetrics wants (N, C); fold any leading dims into N.
                collection.update(
                    c_pred.reshape(-1, c_pred.shape[-1]),
                    c_target.reshape(-1).long(),
                )
            else:  # continuous
                collection.update(c_pred, c_target)

    def compute(self):
        """Compute all metrics and return as a flat dict."""
        results = {}
        if len(self.binary):
            results.update(self.binary.compute())
        if len(self.categorical):
            results.update(self.categorical.compute())
        if len(self.continuous):
            results.update(self.continuous.compute())
        for collection in self._per_concept.values():
            results.update(collection.compute())
        return results

    def reset(self):
        """Reset all metric state."""
        self.binary.reset()
        self.categorical.reset()
        self.continuous.reset()
        for collection in self._per_concept.values():
            collection.reset()


def _predict_task(model: nn.Module, batch: dict, task: str, evidence: List[str]):
    """The model's prediction of ``task`` on ``batch`` (logits if discrete, values if
    continuous), and the true values of the task as an :class:`AnnotatedTensor`."""
    x, c = batch["inputs"]["x"], batch["concepts"]["c"]
    # the input: an inference engine (mid level) or a high-level model is queried with the
    # input and the evidence; a low-level model is called on the input
    if hasattr(model, "pgm"):
        out = model(
            query=[task],
            evidence={"input": x, **{name: c[name].tensor for name in evidence}}
        )
    else:
        out = model(x)

    # the output: an InferenceOutput holds the logits of a discrete task and the loc (or value)
    # of a continuous one; a low-level model returns the task predictions, or (concepts, task)
    y = c[task]
    if isinstance(out, InferenceOutput):
        quantity = (out.logits if y.annotations.types[0] != "continuous"
                    else resolve_quantity(out.params, None, CONTINUOUS_QUANTITIES, y))
        return quantity[task].tensor, y
    return (out[-1] if isinstance(out, tuple) else out), y


@torch.no_grad()
def compute_cace(
    model: nn.Module,
    layer_name: Optional[str],
    concepts_to_intervene_on: Union[List[str], List[slice], List[int]],
    dataloader: Iterable[dict],
    task: str,
    prob_high: Union[float, torch.Tensor] = 1.0,
    prob_low: Union[float, torch.Tensor] = 0.0,
    evidence: List[str] = [],  # needed for higher levels of the API
) -> torch.Tensor:
    """Causal Concept Effect (CaCE) of the concepts in ``concepts_to_intervene_on`` on the task:
    ``E[y | do(C = prob_high)] - E[y | do(C = prob_low)]``, averaged over the dataloader.

    Goyal et al. (2019), `"Explaining Classifiers with Causal Concept Effect (CaCE)"
    <https://arxiv.org/abs/1907.07165>`_. The expectation of the task is its probability if
    binary, the probability of each state if categorical, its value if continuous.

    The concepts are set in **probability space**, converted to the logits the layer
    outputs with ``torch.logit``:

    - binary (default): ``prob_high=1``, ``prob_low=0``;
    - categorical: one probability per output of the layer, e.g.
      ``prob_high=tensor([0, 0, 1])``, ``prob_low=tensor([1, 0, 0])``.

    Works at every level of the library, as :func:`intervention_curve`:

    - low level: ``model`` is a module called as ``model(x)``, returning the task
      predictions (logits if discrete, values if continuous) or
      ``(concept predictions, task predictions)``; ``layer_name`` names its layer
      to intervene on;
    - mid level: ``model`` is an inference engine, high level: a high-level model.
      Both are queried for the task with the input, and the names in
      ``concepts_to_intervene_on`` locate the layer (``layer_name=None``).

    Args:
        model: The model to evaluate.
        layer_name: Attribute of ``model`` holding the layer to intervene on (low
            level); ``None`` for an inference engine or a high-level model.
        concepts_to_intervene_on: One name, slice or position per concept (see
            :func:`intervention`); all of them are set.
        dataloader: Batches in the library's format,
            ``{"inputs": {"x": x}, "concepts": {"c": c}}``, with ``c`` an
            :class:`AnnotatedTensor` holding the task, whose annotation gives its type.
        task: The task to measure: binary, categorical or continuous.
        prob_high: Probability of the concepts in the *high* regime (default 1.0).
        prob_low: Probability of the concepts in the *low* regime (default 0.0).
        evidence: Concepts observed by an inference engine or a high-level model,
            with values from the batch.

    Returns:
        The CaCE: a scalar if the task is binary or continuous, one value per state if
        categorical.

    Example::

        >>> cace = compute_cace(  # doctest: +SKIP
        ...     model=cbm,
        ...     layer_name="concept_encoder",
        ...     concepts_to_intervene_on=["c1"],
        ...     dataloader=test_loader,
        ...     task="y",
        ... )
    """
    target = getattr(model, layer_name) if layer_name is not None else model
    # do(C = prob), on the logits the layer outputs
    regimes = [DoIntervention(torch.logit(torch.as_tensor(prob, dtype=torch.float), eps=1e-6))
               for prob in (prob_low, prob_high)]
    effect_sum, count = 0.0, 0
    for batch in dataloader:
        expected = []  # the expectation of the task in each example, under do(C = prob_low), do(C = prob_high)
        for strategy in regimes:
            with intervention(
                target=target,
                intervention_strategy=strategy,
                intervention_policy=UniformPolicy(),
                out_concepts_to_intervene_on=concepts_to_intervene_on,
            ):
                y_pred, y = _predict_task(model, batch, task, evidence)
            kind = y.annotations.types[0]
            if kind == "binary":
                y_pred = y_pred.sigmoid()
            elif kind == "categorical":
                y_pred = y_pred.softmax(-1)
            expected.append(y_pred.reshape(-1, y_pred.shape[-1]))  # fold any leading dims into the batch

        # Fill the effect, summed over the batches and averaged at the end
        low, high = expected
        effect_sum = effect_sum + (high - low).sum(0)
        count += len(low)
    if not count:
        raise ValueError("compute_cace: the dataloader yielded no batches.")
    return (effect_sum / count).squeeze()


def intervention_curve(
    model: nn.Module,
    layer_name: Optional[str],
    intervention_strategy: Union[InterventionStrategy, Callable[[dict], InterventionStrategy]],
    intervention_policy: InterventionPolicy,
    concepts_to_intervene_on: Union[List[str], List[slice], List[int]],
    dataloader: Iterable[dict],
    task: str,
    evidence: List[str] = [],  # needed for higher levels of the API
    build_context: Optional[Callable] = None,
) -> List[float]:
    """Task performance after intervening on ``n = 0, 1, ..., K`` of the ``K`` concepts in
    ``concepts_to_intervene_on``, chosen in each example by the policy.

    The first point (``n = 0``) is the model without interventions; the last
    intervenes on every concept. With a :class:`GroundTruthIntervention`, this
    is the intervention curve of Koh et al. (2020). The task is scored by its
    type: accuracy if binary or categorical, mean squared error if continuous.

    Works at every level of the library:

    - low level: ``model`` is a module called as ``model(x)``, returning the task
      predictions (logits if discrete, values if continuous) or
      ``(concept predictions, task predictions)``; ``layer_name`` names its layer
      to intervene on;
    - mid level: ``model`` is an inference engine, high level: a high-level model.
      Both are queried for the task with the input, and the names in
      ``concepts_to_intervene_on`` locate the layer (``layer_name=None``).

    Args:
        model: The model to evaluate.
        layer_name: Attribute of ``model`` holding the layer to intervene on (low
            level); ``None`` for an inference engine or a high-level model.
        intervention_strategy: Gives the new values: an
            :class:`InterventionStrategy`, or a callable building one from the
            batch, e.g. a :class:`GroundTruthIntervention` with the batch's true
            concepts.
        intervention_policy: Scores the concepts; the lowest are intervened on first.
        concepts_to_intervene_on: One name, slice or position per concept (see
            :func:`intervention`); a concept named or given as a slice is
            corrected whole, e.g. every state of a categorical one.
        dataloader: Batches in the library's format,
            ``{"inputs": {"x": x}, "concepts": {"c": c}}``, with ``c`` an
            :class:`AnnotatedTensor` holding the true values by name (the index
            of the true state for a categorical task).
        task: The task to score: binary, categorical or continuous.
        evidence: Concepts observed by an inference engine or a high-level model,
            with values from the batch.
        build_context: As in :func:`intervention`; e.g. the concept gradients that
            :class:`GradientPolicy` needs.

    Returns:
        The score for ``n = 0 .. K``.

    Example::

        >>> accuracy = intervention_curve(  # doctest: +SKIP
        ...     model=cbm,
        ...     layer_name="concept_encoder",
        ...     intervention_strategy=lambda batch: GroundTruthIntervention(true_logits(batch)),
        ...     intervention_policy=UncertaintyInterventionPolicy(),
        ...     concepts_to_intervene_on=["c0", "c1", "c2"],
        ...     dataloader=test_loader,
        ...     task="y",
        ... )
        >>> len(accuracy)  # n = 0 .. 3 intervened concepts  # doctest: +SKIP
        4
    """
    target = getattr(model, layer_name) if layer_name is not None else model
    n_concepts = len(concepts_to_intervene_on)
    scores = []
    for n in range(n_concepts + 1):
        score_sum, count = 0.0, 0
        for batch in dataloader:
            strategy = (intervention_strategy if isinstance(intervention_strategy, InterventionStrategy)
                        else intervention_strategy(batch))
            intervened = nullcontext() if n == 0 else intervention(  # n = 0: the baseline, no intervention
                target=target,
                intervention_strategy=strategy,
                intervention_policy=intervention_policy,
                out_concepts_to_intervene_on=concepts_to_intervene_on,
                # all K concepts are candidates and the policy picks n of them: intervention() replaces
                # 1 + floor(quantile * (K - 1)) candidates, so this quantile gives n
                quantile=1.0 if n == n_concepts else (n - 0.5) / (n_concepts - 1),
                build_context=build_context,
            )
            with torch.no_grad(), intervened:
                y_pred, y = _predict_task(model, batch, task, evidence)
            kind = y.annotations.types[0]
            y = y.tensor

            # Fill the score, summed over the batches and averaged at the end
            if kind == "binary":  # accuracy: predicted true where the logit is positive
                per_example = ((y_pred > 0).float() == y).float()
            elif kind == "categorical":  # accuracy: the predicted state has the largest logit
                per_example = (y_pred.argmax(-1, keepdim=True) == y.long()).float()
            else:  # continuous: squared error
                per_example = (y_pred - y) ** 2
            score_sum += per_example.sum().item()
            count += per_example.numel()
        scores.append(score_sum / count)
    return scores
