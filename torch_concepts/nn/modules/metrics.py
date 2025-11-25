"""
Metrics module for concept-based model evaluation.

This module provides custom metrics for evaluating concept-based models,
including causal effect metrics and concept accuracy measures.
"""
from typing import Optional, Union, List
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
from yaml import warnings

from ...annotations import Annotations, AxisAnnotation
from ...nn.modules.utils import GroupConfig
from ...nn.modules.utils import check_collection, get_concept_groups


class ConceptMetrics(nn.Module):
    """Metrics module for concept-based models.
    
    Organizes and manages metrics for different concept types (binary, categorical, 
    continuous) with support for both summary metrics (aggregated across all concepts 
    of a type) and per-concept metrics (individual tracking per concept).
    
    Args:
        annotations (Annotations): Concept annotations with metadata.
        fn_collection (GroupConfig): Metric configurations organized by concept type.
            Each metric can be specified in three ways:
            1. Pre-instantiated: `metric_instance` (e.g., BinaryAccuracy())
            2. Class with user kwargs: `(MetricClass, {'kwarg': value})`
            3. Class only: `MetricClass` (concept-specific params added automatically)
        summary_metrics (bool): Whether to compute summary metrics. Default: True.
        perconcept_metrics (Union[bool, List[str]]): Whether to compute per-concept 
            metrics. If True, computes for all concepts. If list, computes only for 
            specified concept names. Default: False.
            
    Example:
        >>> from torch_concepts.nn.modules import ConceptMetrics, GroupConfig
        >>> import torchmetrics
        >>> 
        >>> # Three ways to specify metrics:
        >>> metrics = ConceptMetrics(
        ...     annotations=concept_annotations,
        ...     fn_collection=GroupConfig(
        ...         binary={
        ...             # 1. Pre-instantiated
        ...             'accuracy': torchmetrics.classification.BinaryAccuracy(),
        ...             # 2. Class + user kwargs (average='macro')
        ...             'f1': (torchmetrics.classification.BinaryF1Score, {'average': 'macro'})
        ...         },
        ...         categorical={
        ...             # 3. Class only (num_classes will be added automatically)
        ...             'accuracy': torchmetrics.classification.MulticlassAccuracy
        ...         }
        ...     ),
        ...     summary_metrics=True,
        ...     perconcept_metrics=['concept1', 'concept2']
        ... )
        >>> 
        >>> # Update metrics during training
        >>> metrics.update(predictions, targets, split='train')
        >>> 
        >>> # Compute metrics at epoch end
        >>> train_metrics = metrics.compute('train')
        >>> metrics.reset('train')
    """
    
    def __init__(
        self,
        annotations: Annotations,
        fn_collection: GroupConfig,
        summary_metrics: bool = True,
        perconcept_metrics: Union[bool, List[str]] = False
    ):
        super().__init__()
        
        self.summary_metrics = summary_metrics
        self.perconcept_metrics = perconcept_metrics
        
        # Extract and validate annotations
        annotations = annotations.get_axis_annotation(axis=1)
        self.concept_annotations = annotations
        self.concept_names = annotations.labels
        self.n_concepts = len(self.concept_names)
        self.cardinalities = annotations.cardinalities
        self.metadata = annotations.metadata
        self.types = [self.metadata[name]['type'] for name in self.concept_names]
        
        # Get concept groups
        self.groups = get_concept_groups(annotations)
        
        # Validate that continuous concepts are not used
        if self.groups['continuous_labels']:
            raise NotImplementedError(
                f"Continuous concepts are not yet supported. "
                f"Found continuous concepts: {self.groups['continuous_labels']}."
            )
        
        # Validate and filter metrics configuration
        self.fn_collection = check_collection(annotations, fn_collection, 'metrics')
        
        # Pre-compute max cardinality for categorical concepts
        if self.fn_collection.get('categorical'):
            self.max_card = max([self.cardinalities[i] 
                                for i in self.groups['categorical_idx']])
        
        # Setup metric collections
        self._setup_metric_collections()
    
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
                f"metrics={{{metrics_str}}}, summary={self.summary_metrics}, "
                f"perconcept={self.perconcept_metrics})")
    
    @staticmethod
    def _clone_metric(metric):
        """Clone and reset a metric for independent tracking across splits."""
        metric = metric.clone()
        metric.reset()
        return metric
    
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
            # Already instantiated
            return metric_spec
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
    
    def _setup_metric_collections(self):
        """Setup MetricCollections for train/val/test splits.
        
        Creates metric collections with appropriate prefixes and cloned metrics
        for each split to ensure independent tracking.
        """
        # Build dictionary of all metrics (summary + per-concept)
        all_metrics = {}
        
        # Add summary metrics
        if self.summary_metrics:
            if self.fn_collection.get('binary'):
                for metric_name, metric_spec in self.fn_collection['binary'].items():
                    key = f"SUMMARY-binary_{metric_name}"
                    all_metrics[key] = self._instantiate_metric(metric_spec)
            
            if self.fn_collection.get('categorical'):
                for metric_name, metric_spec in self.fn_collection['categorical'].items():
                    key = f"SUMMARY-categorical_{metric_name}"
                    # Add num_classes for categorical summary metrics
                    all_metrics[key] = self._instantiate_metric(
                        metric_spec, 
                        concept_specific_kwargs={'num_classes': self.max_card}
                    )
            
            if self.fn_collection.get('continuous'):
                for metric_name, metric_spec in self.fn_collection['continuous'].items():
                    key = f"SUMMARY-continuous_{metric_name}"
                    all_metrics[key] = self._instantiate_metric(metric_spec)
        
        # Add per-concept metrics
        if self.perconcept_metrics:
            # Determine which concepts to track
            if isinstance(self.perconcept_metrics, bool):
                concepts_to_trace = self.concept_names
            elif isinstance(self.perconcept_metrics, list):
                concepts_to_trace = self.perconcept_metrics
            else:
                raise ValueError(
                    "perconcept_metrics must be either a bool or a list of concept names."
                )
            
            for concept_name in concepts_to_trace:
                c_idx = self.concept_names.index(concept_name)
                c_type = self.types[c_idx]
                card = self.cardinalities[c_idx]
                
                # Get the appropriate metrics config for this concept type
                if c_type == 'discrete' and card == 1:
                    metrics_dict = self.fn_collection.get('binary', {})
                    concept_kwargs = {}
                elif c_type == 'discrete' and card > 1:
                    metrics_dict = self.fn_collection.get('categorical', {})
                    concept_kwargs = {'num_classes': card}
                elif c_type == 'continuous':
                    metrics_dict = self.fn_collection.get('continuous', {})
                    concept_kwargs = {}
                else:
                    metrics_dict = {}
                    concept_kwargs = {}
                
                # Add metrics for this concept
                for metric_name, metric_spec in metrics_dict.items():
                    key = f"{concept_name}_{metric_name}"
                    all_metrics[key] = self._instantiate_metric(
                        metric_spec, 
                        concept_specific_kwargs=concept_kwargs
                    )
        
        # Create MetricCollections for each split with cloned metrics
        self.train_metrics = MetricCollection(
            metrics={k: self._clone_metric(m) for k, m in all_metrics.items()},
            prefix="train/"
        ) if all_metrics else MetricCollection({})
        
        self.val_metrics = MetricCollection(
            metrics={k: self._clone_metric(m) for k, m in all_metrics.items()},
            prefix="val/"
        ) if all_metrics else MetricCollection({})
        
        self.test_metrics = MetricCollection(
            metrics={k: self._clone_metric(m) for k, m in all_metrics.items()},
            prefix="test/"
        ) if all_metrics else MetricCollection({})
    
    def get(self, key: str, default=None):
        """Get a metric collection by key (dict-like interface).
        
        Args:
            key (str): Collection key ('train_metrics', 'val_metrics', 'test_metrics').
            default: Default value to return if key not found.
            
        Returns:
            MetricCollection or default value.
        """
        collections = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics
        }
        return collections.get(key, default)
    
    def _get_collection(self, split: str) -> MetricCollection:
        """Get the metric collection for a specific split.
        
        Args:
            split (str): One of 'train', 'val', or 'test'.
            
        Returns:
            MetricCollection: The collection for the specified split.
        """
        if split == 'train':
            return self.train_metrics
        elif split in ['val', 'validation']:
            return self.val_metrics
        elif split == 'test':
            return self.test_metrics
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")
    
    def update(self, input: torch.Tensor, target: torch.Tensor, split: str = 'train'):
        """Update metrics with predictions and targets.
        
        Args:
            input (torch.Tensor): Model predictions (endogenous or values).
            target (torch.Tensor): Ground truth labels/values.
            split (str): Which split to update ('train', 'val', or 'test').
        """
        metric_collection = self._get_collection(split)
        
        for key in metric_collection:
            # Update summary metrics
            if self.summary_metrics:
                if 'SUMMARY-binary_' in key and self.groups['binary_labels']:
                    binary_input = input[:, self.groups['binary_endogenous_idx']]
                    binary_target = target[:, self.groups['binary_idx']].float()
                    metric_collection[key].update(binary_input, binary_target)
                    continue
                
                elif 'SUMMARY-categorical_' in key and self.groups['categorical_labels']:
                    # Pad and stack categorical endogenous
                    split_tuple = torch.split(
                        input[:, self.groups['categorical_endogenous_idx']], 
                        [self.cardinalities[i] for i in self.groups['categorical_idx']], 
                        dim=1
                    )
                    padded_endogenous = [
                        nn.functional.pad(
                            endogenous, 
                            (0, self.max_card - endogenous.shape[1]), 
                            value=float('-inf')
                        ) for endogenous in split_tuple
                    ]
                    cat_input = torch.cat(padded_endogenous, dim=0)
                    cat_target = target[:, self.groups['categorical_idx']].T.reshape(-1).long()
                    metric_collection[key].update(cat_input, cat_target)
                    continue
                
                elif 'SUMMARY-continuous_' in key and self.groups['continuous_labels']:
                    raise NotImplementedError("Continuous concepts not yet implemented.")
            
            # Update per-concept metrics
            if self.perconcept_metrics:
                # Extract concept name from key
                key_noprefix = _remove_prefix(key, prefix=metric_collection.prefix)
                concept_name = '_'.join(key_noprefix.split('_')[:-1])
                if concept_name not in self.concept_names:
                    concept_name = key_noprefix.split('_')[0]
                
                endogenous_idx = self.concept_annotations.get_endogenous_idx([concept_name])
                c_idx = self.concept_annotations.get_index(concept_name)
                c_type = self.types[c_idx]
                card = self.cardinalities[c_idx]
                
                if c_type == 'discrete' and card == 1:
                    metric_collection[key].update(
                        input[:, endogenous_idx], 
                        target[:, c_idx:c_idx+1].float()
                    )
                elif c_type == 'discrete' and card > 1:
                    metric_collection[key].update(
                        input[:, endogenous_idx], 
                        target[:, c_idx].long()
                    )
                elif c_type == 'continuous':
                    metric_collection[key].update(
                        input[:, endogenous_idx], 
                        target[:, c_idx:c_idx+1]
                    )
                else:
                    raise ValueError(f"ConceptMetrics.update(): Unknown concept \
                                     type '{c_type}' for concept '{concept_name}'.")
    
    def compute(self, split: str = 'train'):
        """Compute accumulated metrics for a split.
        
        Args:
            split (str): Which split to compute ('train', 'val', or 'test').
            
        Returns:
            dict: Dictionary of computed metric values.
        """
        metric_collection = self._get_collection(split)
        return metric_collection.compute()
    
    def reset(self, split: Optional[str] = None):
        """Reset metrics for one or all splits.
        
        Args:
            split (Optional[str]): Which split to reset ('train', 'val', 'test'), 
                or None to reset all splits.
        """
        if split is None:
            self.train_metrics.reset()
            self.val_metrics.reset()
            self.test_metrics.reset()
        else:
            metric_collection = self._get_collection(split)
            metric_collection.reset()


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
