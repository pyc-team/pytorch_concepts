"""PyTorch Lightning training engine for concept-based models.

This module provides the Predictor class, which orchestrates the training, 
validation, and testing of concept-based models. It handles:
- Loss computation with type-aware losses (binary/categorical/continuous)
- Metric tracking (summary and per-concept)
- Optimizer and scheduler configuration
- Batch preprocessing and transformations
- Concept interventions (experimental)
"""

from typing import Optional, Mapping, Type, Callable, Union
from abc import abstractmethod

import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from torch_concepts import Annotations
from torch_concepts.nn.modules.utils import check_collection, get_concept_groups
from torch_concepts.utils import add_distribution_to_annotations, instantiate_from_string


class BaseLearner(pl.LightningModule):
    def __init__(self,
                loss: nn.Module,
                metrics: Mapping,
                annotations: Annotations,
                variable_distributions: Mapping,
                optim_class: Type,
                optim_kwargs: Mapping,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,  
                preprocess_inputs: Optional[bool] = False,
                scale_concepts: Optional[bool] = False,
                enable_summary_metrics: Optional[bool] = True,
                enable_perconcept_metrics: Optional[Union[bool, list]] = False,
                **kwargs
                ):
        
        super(BaseLearner, self).__init__(**kwargs)

        self.loss_fn = loss(annotations=annotations)

        # transforms
        self.preprocess_inputs = preprocess_inputs
        self.scale_concepts = scale_concepts

        # metrics configuration
        self.enable_summary_metrics = enable_summary_metrics
        self.enable_perconcept_metrics = enable_perconcept_metrics

        # optimizer and scheduler
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()

        # Add distribution information to annotations metadata
        annotations = add_distribution_to_annotations(
            annotations, variable_distributions
        )
        # concept info
        self.concept_annotations = annotations.get_axis_annotation(1)
        self.metadata = self.concept_annotations.metadata
        self.concept_names = self.concept_annotations.labels
        self.n_concepts = len(self.concept_names)
        self.types = [self.metadata[name]['type'] for name in self.concept_names]

        self.groups = get_concept_groups(self.concept_annotations)

        # Setup and instantiate metrics
        self._setup_metrics(metrics)

    def __repr__(self):
        return "{}(model={}, n_concepts={}, optimizer={}, scheduler={})" \
            .format(self.__class__.__name__,
                    self.n_concepts,
                    self.optim_class.__name__,
                    self.scheduler_class.__name__ if self.scheduler_class else None)

    @staticmethod
    def _check_metric(metric):
        """Clone and reset a metric for independent tracking across splits.
        
        Args:
            metric: TorchMetrics metric instance.
            
        Returns:
            Cloned and reset metric ready for train/val/test collection.
        """
        metric = metric.clone()
        metric.reset()
        return metric

    def _setup_metrics(self, metrics_config: Mapping):
        """Setup and instantiate metrics with summary and/or per-concept tracking.
        
        Creates two types of metrics:
        1. Summary metrics: Aggregated over all concepts of each type 
           (keys: 'SUMMARY-binary_accuracy', etc.)
        2. Per-concept metrics: Individual metrics for specified concepts 
           (keys: 'age_accuracy', 'gender_accuracy', etc.)
        
        Args:
            metrics_config (Mapping): Nested dict with same structure as loss_config.
        """
        if metrics_config is None:
            metrics_config = {}
        
        # Validate and extract needed metrics
        binary_metrics_cfg, categorical_metrics_cfg, continuous_metrics_cfg = check_collection(
            self.concept_annotations, metrics_config, 'metrics'
        )
        
        # Initialize metric storage
        summary_metrics = {}
        perconcept_metrics = {}
        
        # Setup summary metrics (one per type group)
        if self.enable_summary_metrics:
            if binary_metrics_cfg:
                summary_metrics['binary'] = self._instantiate_metric_dict(binary_metrics_cfg)
            
            if categorical_metrics_cfg:
                # For categorical, we'll average over individual concept metrics
                self.max_card = max([self.concept_annotations.cardinalities[i] 
                                     for i in self.groups['categorical_concepts']])
                summary_metrics['categorical'] = self._instantiate_metric_dict(
                    categorical_metrics_cfg, 
                    num_classes=self.max_card
                )
            
            if continuous_metrics_cfg:
                summary_metrics['continuous'] = self._instantiate_metric_dict(continuous_metrics_cfg)
        
        # Setup per-concept metrics (one per concept)
        if self.enable_perconcept_metrics:
            if isinstance(self.enable_perconcept_metrics, bool):
                concepts_to_trace = self.concept_names
            elif isinstance(self.enable_perconcept_metrics, list):
                concepts_to_trace = self.enable_perconcept_metrics
            else:
                raise ValueError("enable_perconcept_metrics must be either a bool or a list of concept names.")
            for concept_name in concepts_to_trace:
                c_id = self.concept_names.index(concept_name)
                c_type = self.types[c_id]
                card = self.concept_annotations.cardinalities[c_id]
                
                # Select the appropriate metrics config for this concept
                if c_type == 'discrete' and card == 1:
                    metrics_cfg = binary_metrics_cfg
                elif c_type == 'discrete' and card > 1:
                    metrics_cfg = categorical_metrics_cfg
                elif c_type == 'continuous':
                    metrics_cfg = continuous_metrics_cfg
                else:
                    metrics_cfg = None
                
                # Instantiate metrics for this concept
                concept_metric_dict = {}
                if metrics_cfg is not None:
                    for metric_name, metric_dict in metrics_cfg.items():
                        kwargs = metric_dict.get('kwargs', {})
                        if c_type == 'discrete' and card > 1:
                            kwargs['num_classes'] = card
                        concept_metric_dict[metric_name] = instantiate_from_string(metric_dict['path'], **kwargs)
                
                perconcept_metrics[concept_name] = concept_metric_dict
        
        # Create metric collections for train/val/test
        self._set_metrics(summary_metrics, perconcept_metrics)
    
    def _instantiate_metric_dict(self, metrics_cfg: Mapping, num_classes: int = None) -> dict:
        """Instantiate a dictionary of metrics from configuration.
        
        Args:
            metrics_cfg (Mapping): Dict of metric configs with 'path' and 'kwargs'.
            num_classes (int, optional): Number of classes for categorical metrics. 
                If provided, overrides kwargs['num_classes'].
                
        Returns:
            dict: Instantiated metrics keyed by metric name.
        """
        if not isinstance(metrics_cfg, dict):
            return {}
        
        metrics = {}
        for metric_name, metric_path in metrics_cfg.items():
            kwargs = metric_path.get('kwargs', {})
            if num_classes is not None:
                kwargs['num_classes'] = num_classes
            metrics[metric_name] = instantiate_from_string(metric_path['path'], **kwargs)
        return metrics

    def _set_metrics(self, summary_metrics: Mapping = None, perconcept_metrics: Mapping = None):
        """Create MetricCollections for train/val/test splits.
        
        Combines summary and per-concept metrics into MetricCollections with 
        appropriate prefixes ('train/', 'val/', 'test/').
        
        Args:
            summary_metrics (Mapping, optional): Dict of summary metrics by type.
            perconcept_metrics (Mapping, optional): Dict of per-concept metrics.
        """
        all_metrics = {}
        
        # Add summary metrics
        if summary_metrics:
            for group_name, metric_dict in summary_metrics.items():
                for metric_name, metric in metric_dict.items():
                    key = f"SUMMARY-{group_name}_{metric_name}"
                    all_metrics[key] = metric
        
        # Add per-concept metrics
        if perconcept_metrics:
            for concept_name, metric_dict in perconcept_metrics.items():
                for metric_name, metric in metric_dict.items():
                    key = f"{concept_name}_{metric_name}"
                    all_metrics[key] = metric
        
        # Create collections
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in all_metrics.items()},
            prefix="train/"
        ) if all_metrics else MetricCollection({})
        
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in all_metrics.items()},
            prefix="val/"
        ) if all_metrics else MetricCollection({})
        
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in all_metrics.items()},
            prefix="test/"
        ) if all_metrics else MetricCollection({})

    def _apply_fn_by_type(self, 
                         c_hat: torch.Tensor, 
                         c_true: torch.Tensor,
                         binary_fn: Optional[Callable],
                         categorical_fn: Optional[Callable],
                         continuous_fn: Optional[Callable]) -> Union[torch.Tensor, None]:
        """Apply metric functions to concept groups by type.
        
        Slices predictions and targets by concept type and applies the 
        appropriate function to each group. Handles padding for categorical
        concepts with varying cardinalities.
        
        Args:
            c_hat (torch.Tensor): Predicted concepts (logits or values).
            c_true (torch.Tensor): Ground truth concepts.
            binary_fn (Optional[Callable]): Function for binary concepts 
                (metric.update).
            categorical_fn (Optional[Callable]): Function for categorical concepts.
            continuous_fn (Optional[Callable]): Function for continuous concepts.
            
        Returns:
            Union[torch.Tensor, None]: Scalar loss tensor if is_loss=True, 
                else None (metrics updated in-place).
                
        Note:
            For categorical concepts, logits are padded to max_card and stacked
            for batch processing. This is a known performance bottleneck (FIXME).
        """

        if binary_fn:
            c_hat_binary = c_hat[:, self.groups['binary_logits']]
            c_true_binary = c_true[:, self.groups['binary_concepts']].float()
            binary_fn.update(c_hat_binary, c_true_binary)

        if categorical_fn:
            # Pad all tensors to max cardinality and stack
            # FIXME: optimize this operation, could this for loop be avoided?
            split_tuple = torch.split(c_hat[:, self.groups['categorical_logits']], 
                                      [self.concept_annotations.cardinalities[i] 
                                       for i in self.groups['categorical_concepts']], dim=1)
            padded_logits = [
                torch.nn.functional.pad(logits, (0, self.max_card - logits.shape[1]), value=float('-inf'))
                for logits in split_tuple
            ]
            c_hat_group = torch.cat(padded_logits, dim=0)
            c_true_group = c_true[:, self.groups['categorical_concepts']].T.reshape(-1).long()
            
            categorical_fn.update(c_hat_group, c_true_group)

        if continuous_fn:
            # TODO: verify correctness
            c_hat_continuous = c_hat[:, self.groups['continuous_logits']]
            c_true_continuous = c_true[:, self.groups['continuous_concepts']]
            continuous_fn.update(c_hat_continuous, c_true_continuous)

    
    def update_metrics(self, in_metric_dict: Mapping, 
                       metric_collection: MetricCollection):
        """Update both summary and per-concept metrics.
        
        Iterates through the metric collection and updates each metric with
        the appropriate slice of predictions and targets based on metric type
        (summary vs per-concept) and concept type (binary/categorical/continuous).
        
        Args:
            c_hat (torch.Tensor): Predicted concepts.
            c_true (torch.Tensor): Ground truth concepts.
            metric_collection (MetricCollection): Collection to update (train/val/test).
        """
        c_hat = in_metric_dict['input']
        c_true = in_metric_dict['target']
        
        for key in metric_collection:

            # Update summary metrics (compute metrics relative to each group)
            if self.enable_summary_metrics:
                if 'SUMMARY-binary_' in key and self.groups['binary_concepts']:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=metric_collection[key],
                        categorical_fn=None,
                        continuous_fn=None
                    )
                    continue
                
                elif 'SUMMARY-categorical_' in key and self.groups['categorical_concepts']:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=None,
                        categorical_fn=metric_collection[key],
                        continuous_fn=None
                    )
                    continue
                
                elif 'SUMMARY-continuous_' in key and self.groups['continuous_concepts']:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=None,
                        categorical_fn=None,
                        continuous_fn=metric_collection[key]
                    )
                    continue

            # Update per-concept metrics
            if self.enable_perconcept_metrics:
                # Extract concept name from key
                key_noprefix = _remove_prefix(key, prefix=metric_collection.prefix)
                concept_name = '_'.join(key_noprefix.split('_')[:-1])  # Handle multi-word concept names
                if concept_name not in self.concept_names:
                    concept_name = key_noprefix.split('_')[0]  # Fallback to simple split
                
                c_id = self.concept_names.index(concept_name)
                c_type = self.types[c_id]
                card = self.concept_annotations.cardinalities[c_id]

                start_idx = self.groups['cumulative_indices'][c_id]
                end_idx = self.groups['cumulative_indices'][c_id + 1]

                if c_type == 'discrete' and card == 1:
                    metric_collection[key].update(c_hat[:, start_idx:end_idx], 
                                                  c_true[:, c_id:c_id+1].float())
                elif c_type == 'discrete' and card > 1:
                    # Extract logits for this categorical concept
                    metric_collection[key].update(c_hat[:, start_idx:end_idx], 
                                                  c_true[:, c_id].long())
                elif c_type == 'continuous':
                    metric_collection[key].update(c_hat[:, start_idx:end_idx], 
                                                  c_true[:, c_id:c_id+1])
 
    def log_metrics(self, metrics, **kwargs):
        """Log metrics to logger (W&B) at epoch end.
        
        Args:
            metrics: MetricCollection or dict of metrics to log.
            **kwargs: Additional arguments passed to self.log_dict.
        """
        self.log_dict(metrics, 
                      on_step=False, 
                      on_epoch=True, 
                      logger=True, 
                      prog_bar=False, 
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        """Log loss to logger and progress bar at epoch end.
        
        Args:
            name (str): Loss name prefix (e.g., 'train', 'val', 'test').
            loss (torch.Tensor): Loss value to log.
            **kwargs: Additional arguments passed to self.log.
        """
        self.log(name + "_loss",
                 loss.detach(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=True,
                 **kwargs)

    def unpack_batch(self, batch):
        """Extract inputs, concepts, and transforms from batch dict.
        can be overridden by model-specific preprocessing.
        
        Args:
            batch (dict): Batch with 'inputs', 'concepts', and optional 'transform'.
            
        Returns:
            Tuple: (inputs, concepts, transforms) after model-specific preprocessing.
        """
        inputs = batch['inputs']
        concepts = batch['concepts']
        transforms = batch.get('transforms', {})
        return inputs, concepts, transforms

    @abstractmethod
    def training_step(self, batch):
        """Training step called by PyTorch Lightning.
        
        Args:
            batch (dict): Training batch.
            
        Returns:
            torch.Tensor: Training loss.
        """
        pass

    @abstractmethod
    def validation_step(self, batch):
        """Validation step called by PyTorch Lightning.
        
        Args:
            batch (dict): Validation batch.
            
        Returns:
            torch.Tensor: Validation loss.
        """
        pass
    
    @abstractmethod    
    def test_step(self, batch):
        """Test step called by PyTorch Lightning.
        
        Args:
            batch (dict): Test batch.
            
        Returns:
            torch.Tensor: Test loss.
        """
        pass

    # TODO: custom predict_step?
    # @abstractmethod
    # def predict_step(self, batch):
    #     pass

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler.
        
        Called by PyTorch Lightning to setup optimization.
        
        Returns:
            dict: Configuration with 'optimizer' and optionally 'lr_scheduler' 
                and 'monitor' keys.
                
        Example:
            >>> # With scheduler monitoring validation loss
            >>> predictor = Predictor(
            ...     ...,
            ...     optim_class=torch.optim.Adam,
            ...     optim_kwargs={'lr': 0.001},
            ...     scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            ...     scheduler_kwargs={'mode': 'min', 'patience': 5, 'monitor': 'val_loss'}
            ... )
        """
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg
 