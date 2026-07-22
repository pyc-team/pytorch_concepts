
from typing import Optional, Union, List, Tuple
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from copy import deepcopy

from ...annotations import Annotations
from .outputs import CONTINUOUS_QUANTITIES, ModelOutput
from .utils import GroupConfig, check_collection


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
        fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
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
            if tensor is not None and name in tensor.annotation.label_to_index:
                # Hand torchmetrics a plain tensor; the annotation is no longer
                # needed and would make each internal torch.* op pay the
                # __torch_function__ cost.
                return tensor[name].tensor
        raise KeyError(
            f"ConceptMetrics: no {' or '.join(quantities)} reported for {concept_type} "
            f"concept {name!r}; the output carries {tuple(out.params)}."
        )

    def _prepare_categorical(self, cat_logits, cat_target):
        """Pad and stack categorical logits/targets for the summary metric.

        ``cat_logits`` (logit-space) and ``cat_target`` (concept-space) are the
        categorical slices from :meth:`AnnotatedTensor.split_by_type`, already in
        the same concept order; per-concept widths come from ``cat_logits``'s own
        annotation.
        """
        # Unwrap to plain tensors up front: past this point the annotation has
        # served its purpose (concept ordering), and every torch.* op below on an
        # AnnotatedTensor would otherwise re-enter its __torch_function__ hook.
        cards = list(cat_logits.annotation.cardinalities)
        split_tuple = torch.split(cat_logits.tensor, cards, dim=1)
        padded_logits = [
            nn.functional.pad(
                logits,
                (0, self.max_card - logits.shape[1]),
                value=float('-inf')
            ) for logits in split_tuple
        ]
        cat_pred = torch.cat(padded_logits, dim=0)
        cat_target = cat_target.tensor.T.reshape(-1).long()
        return cat_pred, cat_target

    def update(self, preds, target: torch.Tensor = None):
        """Update metrics by routing predictions to the correct type collection.

        Summary metrics receive aggregated data for all concepts of a type.
        Per-concept metrics receive individual concept data.

        Args:
            preds: A ``ModelOutput`` or a single :class:`AnnotatedTensor` of
                discrete predictions.
            target: Annotated concept-space ground truth. Defaults to
                ``preds.target`` when *preds* is a ``ModelOutput``; required
                otherwise.
        """
        # A ModelOutput carries one AnnotatedTensor per quantity (logits/probs for
        # discrete concepts, loc/scale for continuous); each type is scored on the
        # quantity that represents it. A bare tensor is taken as the discrete params.
        if isinstance(preds, ModelOutput):
            out = preds
            target = target if target is not None else out.target
        else:
            out = ModelOutput()
            out.logits = preds

        discrete = out.logits if out.logits is not None else out.probs
        # `loc` for a Normal, `value` for a Delta (a deterministic point estimate) —
        # mirrors the discrete fallback above; unlike *_param on ConceptLoss, there
        # is no per-instance config here, so both quantities are always tried.
        continuous = out.loc if out.loc is not None else out.value
        any_pred = discrete if discrete is not None else continuous
        if any_pred is None or any_pred.shape[0] == 0:
            return
        disc_by_type = discrete.split_by_type() if discrete is not None else {}

        # Summary metrics — one MetricCollection.update() per type, each scored on
        # the quantity it reports and aligned to the target by concept name.
        if self.summary:
            # Predictions/targets are unwrapped to plain tensors before every
            # torchmetrics update: the annotation is only used to align preds to
            # the target by concept name, and passing an AnnotatedTensor into
            # torchmetrics makes each internal torch.* op re-enter the
            # __torch_function__ unwrap hook (the dominant per-update cost).
            if 'binary' in disc_by_type and len(self.binary):
                p = disc_by_type['binary']
                self.binary.update(p.tensor, target[p.annotation.labels].tensor.float())
            if 'categorical' in disc_by_type and len(self.categorical):
                p = disc_by_type['categorical']
                cat_pred, cat_target = self._prepare_categorical(p, target[p.annotation.labels])
                self.categorical.update(cat_pred, cat_target)
            if continuous is not None and len(self.continuous):
                self.continuous.update(
                    continuous.tensor, target[continuous.annotation.labels].tensor)

        # Per-concept metrics — read each concept from its type's quantity.
        for concept_name, collection in self._per_concept.items():
            c_type = self.types[self.concept_names.index(concept_name)]
            c_pred = self._concept_prediction(out, concept_name, c_type)
            c_target = target[concept_name].tensor
            if c_type == 'binary':
                collection.update(c_pred, c_target.float())
            elif c_type == 'categorical':
                collection.update(c_pred, c_target.reshape(-1).long())
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


@torch.no_grad()
def compute_cace(
    model,
    dataloader,
    source_concept: str,
    target_concept: str,
    prob_high: Union[float, torch.Tensor] = 1.0,
    prob_low: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Compute the Causal Concept Effect of *source_concept* on *target_concept*.

    Runs ``do(source = prob_high)`` vs ``do(source = prob_low)`` over the
    dataloader and returns the mean difference on the target.

    Values are in **probability space** (0–1 for binary concepts).
    They are converted to logits internally via ``torch.logit``.

    * **Binary** (default): ``prob_high=1, prob_low=0``.
    * **Categorical**: pass probability vectors, e.g.
      ``prob_high=tensor([0, 0, 1])``, ``prob_low=tensor([1, 0, 0])``.

    Args:
        model: A high-level concept model (e.g.
            :class:`~torch_concepts.nn.ConceptBottleneckModel`).
        dataloader: Iterable yielding batch dicts with
            ``{'inputs': {'x': Tensor}}``.
        source_concept: Concept to intervene on.
        target_concept: Concept whose prediction is measured.
        prob_high: Probability for the *high* regime (default 1.0).
        prob_low: Probability for the *low* regime (default 0.0).

    Returns:
        Scalar tensor with the CaCE score.

    Example::

        >>> cace = compute_cace(  # doctest: +SKIP
        ...     model=cbm,
        ...     dataloader=test_loader,
        ...     source_concept="c1",
        ...     target_concept="task",
        ... )
    """
    from ..modules.low.inference.intervention import DoIntervention, intervention
    from ..modules.low.policy.uniform import UniformPolicy
    from ...nn.functional import cace_score

    cpds = model.model.probabilistic_model.parametric_cpds
    was_training = model.training
    model.eval()

    if not any(True for _ in dataloader):
        if was_training:
            model.train()
        raise ValueError("Dataloader yielded no batches.")

    # Convert probabilities → logits (the intervention operates in logit space)
    eps = 1e-6
    if not torch.is_tensor(prob_high):
        prob_high = torch.tensor(prob_high)
    if not torch.is_tensor(prob_low):
        prob_low = torch.tensor(prob_low)
    logit_high = torch.logit(prob_high.float(), eps=eps)
    logit_low = torch.logit(prob_low.float(), eps=eps)

    strategy_high = DoIntervention(model=cpds, constants=logit_high)
    strategy_low = DoIntervention(model=cpds, constants=logit_low)

    all_high, all_low = [], []
    for batch in dataloader:
        x = batch["inputs"]["x"]

        with intervention(
            policies=UniformPolicy(out_concepts=1),
            strategies=strategy_high,
            target_concepts=[source_concept],
        ):
            out_high = model(x=x, query=[target_concept])

        with intervention(
            policies=UniformPolicy(out_concepts=1),
            strategies=strategy_low,
            target_concepts=[source_concept],
        ):
            out_low = model(x=x, query=[target_concept])

        all_high.append(out_high.probs)
        all_low.append(out_low.probs)

    if was_training:
        model.train()

    return cace_score(torch.cat(all_low), torch.cat(all_high)).squeeze()
