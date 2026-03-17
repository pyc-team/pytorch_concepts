
from typing import Optional, Union, List, Tuple
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from copy import deepcopy

from ...annotations import Annotations
from ...nn.modules.utils import GroupConfig
from ...nn.modules.utils import check_collection


def clone_metric(metric):
    """Clone and reset a metric for independent tracking across splits."""
    metric = metric.clone()
    metric.reset()
    return metric


class ConceptMetrics(nn.Module):
    """Type-aware metric manager for concept-based models.

    Automatically routes predictions to the correct metrics based on concept
    type (binary / categorical) as defined in the annotations. Supports
    summary metrics (aggregated per type) and per-concept metrics, with
    independent state for each data split (train / val / test).

    Args:
        annotations (Annotations): Concept annotations (axis 1) with labels,
            cardinalities, and metadata specifying ``'discrete'`` types.
        binary: Metric specs for binary concepts (cardinality 1).
        categorical: Metric specs for categorical concepts (cardinality > 1).
        continuous: Metric specs for continuous concepts (not yet supported).
        summary (bool): Compute summary metrics aggregated across all
            concepts of each type.  Default ``True``.
        per_concept (bool | list[str]): ``False`` (default) disables
            per-concept tracking; ``True`` tracks every concept; a list
            of names tracks only those concepts.

    Each metric spec can be:

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
        metrics.update(preds, targets, split="train")
        results = metrics.compute("train")   # {"train/SUMMARY-binary_accuracy": ...}
        metrics.reset("train")
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
        
        # Extract and validate annotations
        annotations = annotations.get_axis_annotation(axis=1)
        self.concept_annotations = annotations
        self.concept_names = annotations.labels
        self.n_concepts = len(self.concept_names)
        self.cardinalities = annotations.cardinalities
        self.metadata = annotations.metadata
        self.types = [self.metadata[name]['type'] for name in self.concept_names]
        
        # Use cached type_groups from AxisAnnotation
        self.groups = annotations.type_groups
        
        # Validate that continuous concepts are not used
        if self.groups['continuous']['labels']:
            raise NotImplementedError(
                f"Continuous concepts are not yet supported. "
                f"Found continuous concepts: {self.groups['continuous']['labels']}."
            )
        
        # Validate and filter metrics configuration
        fn_collection = GroupConfig(binary=binary, categorical=categorical, continuous=continuous)
        self.fn_collection = check_collection(annotations, fn_collection, 'metrics')
        
        # Pre-compute max cardinality for categorical concepts
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
            if c_type == 'discrete' and card == 1:
                for name, spec in self.fn_collection.get('binary', {}).items():
                    concept_metrics[name] = self._instantiate_metric(spec)
            elif c_type == 'discrete' and card > 1:
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
    
    def _prepare_categorical(self, preds, target):
        """Pad and stack categorical logits/targets for summary metrics."""
        cat_concept_idx = self.groups['categorical']['concept_idx']
        split_tuple = torch.split(
            preds[:, self.groups['categorical']['logits_idx']],
            [self.cardinalities[i] for i in cat_concept_idx],
            dim=1
        )
        padded_logits = [
            nn.functional.pad(
                logits,
                (0, self.max_card - logits.shape[1]),
                value=float('-inf')
            ) for logits in split_tuple
        ]
        cat_pred = torch.cat(padded_logits, dim=0)
        cat_target = target[:, cat_concept_idx].T.reshape(-1).long()
        return cat_pred, cat_target
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metrics by routing predictions to the correct type collection.
        
        Summary metrics receive aggregated data for all concepts of a type.
        Per-concept metrics receive individual concept data.
        
        Args:
            preds: Model predictions (logits). Shape ``(batch, logits_dim)``.
            target: Ground truth values. Shape ``(batch, n_concepts)``.
        """
        if preds.shape[0] == 0:
            return
        
        # Summary metrics — one MetricCollection.update() call per type
        if self.summary:
            if self.groups['binary']['labels'] and len(self.binary):
                binary_pred = preds[:, self.groups['binary']['logits_idx']]
                binary_target = target[:, self.groups['binary']['concept_idx']].float()
                self.binary.update(binary_pred, binary_target)
            
            if self.groups['categorical']['labels'] and len(self.categorical):
                cat_pred, cat_target = self._prepare_categorical(preds, target)
                self.categorical.update(cat_pred, cat_target)
            
            if self.groups['continuous']['labels'] and len(self.continuous):
                raise NotImplementedError("Continuous concepts not yet implemented.")
        
        # Per-concept metrics — one MetricCollection.update() call per concept
        for concept_name, collection in self._per_concept.items():
            logits_slice = self.concept_annotations.get_slice(concept_name)
            c_idx = self.concept_annotations.get_index(concept_name)
            c_type = self.types[c_idx]
            card = self.cardinalities[c_idx]
            
            if c_type == 'discrete' and card == 1:
                collection.update(preds[:, logits_slice], target[:, c_idx:c_idx+1].float())
            elif c_type == 'discrete' and card > 1:
                collection.update(preds[:, logits_slice], target[:, c_idx].long())
            elif c_type == 'continuous':
                collection.update(preds[:, logits_slice], target[:, c_idx:c_idx+1])
    
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

# class ConceptCausalEffect(Metric):
#     """
#     Concept Causal Effect (CaCE) metric for measuring causal effects.
#
#     CaCE measures the causal effect between concept pairs or between a concept
#     and the task by comparing predictions under interventions do(C=1) vs do(C=0).
#
#     Note: Currently only works on binary concepts.
#
#     Attributes:
#         preds_do_1 (Tensor): Accumulated predictions under do(C=1).
#         preds_do_0 (Tensor): Accumulated predictions under do(C=0).
#         total (Tensor): Total number of samples processed.
#
#     Example:
#         >>> import torch
#         >>> from torch_concepts.nn.modules.metrics import ConceptCausalEffect
#         >>>
#         >>> # Create metric
#         >>> cace = ConceptCausalEffect()
#         >>>
#         >>> # Update with predictions under interventions
#         >>> preds_do_1 = torch.tensor([[0.1, 0.9], [0.2, 0.8]])  # P(Y|do(C=1))
#         >>> preds_do_0 = torch.tensor([[0.8, 0.2], [0.7, 0.3]])  # P(Y|do(C=0))
#         >>> cace.update(preds_do_1, preds_do_0)
#         >>>
#         >>> # Compute causal effect
#         >>> effect = cace.compute()
#         >>> print(f"Causal effect: {effect:.3f}")
#
#     References:
#         Goyal et al. "Explaining Classifiers with Causal Concept Effect (CaCE)",
#         arXiv 2019. https://arxiv.org/abs/1907.07165
#     """
#     def __init__(self):
#         super().__init__()
#         self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, 
#                preds_do_1: torch.Tensor, 
#                preds_do_0: torch.Tensor):
#         """
#         Update metric state with predictions under interventions.
#
#         Args:
#             preds_do_1: Predictions when intervening C=1, shape (batch_size, n_classes).
#             preds_do_0: Predictions when intervening C=0, shape (batch_size, n_classes).
#         """
#         _check_same_shape(preds_do_1, preds_do_0)
#         # expected value = 1*p(output=1|do(1)) + 0*(1-p(output=1|do(1))
#         self.preds_do_1 += preds_do_1[:,1].sum()
#         # expected value = 1*p(output=1|do(0)) + 0*(1-p(output=1|do(0))
#         self.preds_do_0 += preds_do_0[:,1].sum()
#         self.total += preds_do_1.size()[0]

#     def compute(self):
#         """
#         Compute the Causal Concept Effect (CaCE).
#
#         Returns:
#             torch.Tensor: The average causal effect E[Y|do(C=1)] - E[Y|do(C=0)].
#         """
#         return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float()  / self.total)
