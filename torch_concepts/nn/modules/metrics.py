"""
Metrics module for concept-based model evaluation.

This module provides the :class:`ConceptMetrics` class for evaluating concept-based models
with automatic handling of different concept types (binary, categorical, continuous).
It integrates seamlessly with TorchMetrics and PyTorch Lightning, providing flexible
metric tracking at both aggregate and per-concept levels.

Key Features:
    - Automatic routing of concept predictions to appropriate metrics based on type
    - Summary metrics: aggregated performance across all concepts of each type
    - Per-concept metrics: individual tracking for specific concepts
    - Flexible metric specification: pre-instantiated, class+kwargs, or class-only
    - Independent tracking across train/validation/test splits
    - Integration with PyTorch Lightning training loops

Classes:
    ConceptMetrics: Main metrics manager for concept-based models

Example:
    Basic usage with binary and categorical concepts::

        import torch
        import torchmetrics
        from torch_concepts import Annotations, AxisAnnotation
        from torch_concepts.nn.modules.metrics import ConceptMetrics
        from torch_concepts.nn.modules.utils import GroupConfig

        # Define concept structure
        annotations = Annotations({
            1: AxisAnnotation(
                labels=['is_round', 'is_smooth', 'color'],
                cardinalities=[1, 1, 3],  # binary, binary, categorical
                metadata={
                    'is_round': {'type': 'discrete'},
                    'is_smooth': {'type': 'discrete'},
                    'color': {'type': 'discrete'}
                }
            )
        })

        # Configure metrics
        metrics = ConceptMetrics(
            annotations=annotations,
            fn_collection=GroupConfig(
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy}
            ),
            summary_metrics=True,
            perconcept_metrics=True
        )

        # During training
        predictions = torch.randn(32, 5)  # 2 binary + 3 categorical (endogenous space)
        targets = torch.cat([
            torch.randint(0, 2, (32, 2)),  # binary concepts
            torch.randint(0, 3, (32, 1))   # categorical concept
        ], dim=1)

        metrics.update(preds=predictions, target=targets, split='train')
        results = metrics.compute('train')
        metrics.reset('train')

See Also:
    - :doc:`/guides/using_metrics`: Comprehensive guide to using metrics
    - :doc:`/modules/nn.loss`: Loss functions for concept-based models
    - :class:`torch_concepts.nn.modules.utils.GroupConfig`: Metric configuration helper
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
    """Metrics manager for concept-based models with automatic type-aware routing.
    
    This class organizes and manages metrics for different concept types (binary, 
    categorical, continuous) with support for both summary metrics (aggregated across 
    all concepts of a type) and per-concept metrics (individual tracking per concept).
    
    The class automatically routes predictions to the appropriate metrics based on
    concept types defined in the annotations, handles different metric instantiation
    patterns, and maintains independent metric tracking across train/val/test splits.
    
    Args:
        annotations (Annotations): Concept annotations containing labels, types, and 
            cardinalities. Should include axis 1 (concept axis) with metadata specifying
            concept types as 'discrete' or 'continuous'.
        fn_collection (GroupConfig): Metric configurations organized by concept type 
            ('binary', 'categorical', 'continuous'). Each metric can be specified in 
            three ways:
            
            1. **Pre-instantiated metric**: Pass an already instantiated metric object
               for full control over all parameters.
               
               Example::
               
                   'accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.6)
            
            2. **Class with user kwargs**: Pass a tuple of (MetricClass, kwargs_dict)
               to provide custom parameters while letting ConceptMetrics handle 
               concept-specific parameters like num_classes automatically.
               
               Example::
               
                   'accuracy': (torchmetrics.classification.MulticlassAccuracy, 
                               {'average': 'macro'})
            
            3. **Class only**: Pass just the metric class and let ConceptMetrics handle 
               all instantiation with appropriate concept-specific parameters.
               
               Example::
               
                   'accuracy': torchmetrics.classification.MulticlassAccuracy
                   
        summary_metrics (bool, optional): Whether to compute summary metrics that 
            aggregate performance across all concepts of each type. Defaults to True.
        perconcept_metrics (Union[bool, List[str]], optional): Controls per-concept 
            metric tracking. Options:
            
            - False: No per-concept tracking (default)
            - True: Track all concepts individually  
            - List[str]: Track only the specified concept names
            
    Attributes:
        n_concepts (int): Total number of concepts
        concept_names (Tuple[str]): Names of all concepts
        cardinalities (List[int]): Number of classes for each concept
        summary_metrics (bool): Whether summary metrics are computed
        perconcept_metrics (Union[bool, List[str]]): Per-concept tracking configuration
        train_metrics (MetricCollection): Metrics for training split
        val_metrics (MetricCollection): Metrics for validation split  
        test_metrics (MetricCollection): Metrics for test split
        
    Raises:
        NotImplementedError: If continuous concepts are found (not yet supported)
        ValueError: If metric configuration doesn't match concept types, or if 
            user provides num_classes when it should be set automatically
            
    Example:
        **Basic usage with pre-instantiated metrics**::
        
            import torch
            import torchmetrics
            from torch_concepts import Annotations, AxisAnnotation
            from torch_concepts.nn.modules.metrics import ConceptMetrics
            from torch_concepts.nn.modules.utils import GroupConfig
            
            # Define concept structure
            annotations = Annotations({
                1: AxisAnnotation(
                    labels=('round', 'smooth'),
                    cardinalities=[1, 1],
                    metadata={
                        'round': {'type': 'discrete'},
                        'smooth': {'type': 'discrete'}
                    }
                )
            })
            
            # Create metrics with pre-instantiated objects
            metrics = ConceptMetrics(
                annotations=annotations,
                fn_collection=GroupConfig(
                    binary={
                        'accuracy': torchmetrics.classification.BinaryAccuracy(),
                        'f1': torchmetrics.classification.BinaryF1Score()
                    }
                ),
                summary_metrics=True,
                perconcept_metrics=False
            )
            
            # Simulate training batch
            predictions = torch.randn(32, 2)  # endogenous predictions
            targets = torch.randint(0, 2, (32, 2))  # binary targets
            
            # Update metrics
            metrics.update(pred=predictions, target=targets, split='train')
            
            # Compute at epoch end
            results = metrics.compute('train')
            print(results)  # {'train/SUMMARY-binary_accuracy': ..., 'train/SUMMARY-binary_f1': ...}
            
            # Reset for next epoch
            metrics.reset('train')
            
        **Using class + kwargs for flexible configuration**::
        
            # Mixed concept types with custom metric parameters
            annotations = Annotations({
                1: AxisAnnotation(
                    labels=('binary1', 'binary2', 'category'),
                    cardinalities=[1, 1, 5],
                    metadata={
                        'binary1': {'type': 'discrete'},
                        'binary2': {'type': 'discrete'},
                        'category': {'type': 'discrete'}
                    }
                )
            })
            
            metrics = ConceptMetrics(
                annotations=annotations,
                fn_collection=GroupConfig(
                    binary={
                        # Custom threshold
                        'accuracy': (torchmetrics.classification.BinaryAccuracy, 
                                   {'threshold': 0.6})
                    },
                    categorical={
                        # Custom averaging, num_classes added automatically
                        'accuracy': (torchmetrics.classification.MulticlassAccuracy,
                                   {'average': 'macro'})
                    }
                ),
                summary_metrics=True,
                perconcept_metrics=True  # Track all concepts individually
            )
            
            # Predictions: 2 binary + 5 categorical = 7 dimensions
            predictions = torch.randn(16, 7)
            targets = torch.cat([
                torch.randint(0, 2, (16, 2)),  # binary
                torch.randint(0, 5, (16, 1))   # categorical
            ], dim=1)
            
            metrics.update(pred=predictions, target=targets, split='train')
            results = metrics.compute('train')
            
            # Results include both summary and per-concept metrics:
            # 'train/SUMMARY-binary_accuracy'
            # 'train/SUMMARY-categorical_accuracy'
            # 'train/binary1_accuracy'
            # 'train/binary2_accuracy'
            # 'train/category_accuracy'
            
        **Selective per-concept tracking**::
        
            # Track only specific concepts
            metrics = ConceptMetrics(
                annotations=annotations,
                fn_collection=GroupConfig(
                    binary={'accuracy': torchmetrics.classification.BinaryAccuracy}
                ),
                summary_metrics=True,
                perconcept_metrics=['binary1']  # Only track binary1 individually
            )
            
        **Integration with PyTorch Lightning**::
        
            import pytorch_lightning as pl
            
            class ConceptModel(pl.LightningModule):
                def __init__(self, annotations):
                    super().__init__()
                    self.model = ... # your model
                    self.metrics = ConceptMetrics(
                        annotations=annotations,
                        fn_collection=GroupConfig(
                            binary={'accuracy': torchmetrics.classification.BinaryAccuracy}
                        ),
                        summary_metrics=True
                    )
                    
                def training_step(self, batch, batch_idx):
                    x, concepts = batch
                    preds = self.model(x)
                    
                    # Update metrics
                    self.metrics.update(pred=preds, target=concepts, split='train')
                    return loss
                    
                def on_train_epoch_end(self):
                    # Compute and log metrics
                    metrics_dict = self.metrics.compute('train')
                    self.log_dict(metrics_dict)
                    self.metrics.reset('train')
                    
    Note:
        - Continuous concepts are not yet supported and will raise NotImplementedError
        - For categorical concepts, ConceptMetrics automatically handles padding to
          the maximum cardinality when computing summary metrics
        - User-provided 'num_classes' parameter for categorical metrics will raise
          an error as it's set automatically based on concept cardinalities
        - Each split (train/val/test) maintains independent metric state
        
    See Also:
        - :class:`torch_concepts.nn.modules.utils.GroupConfig`: Configuration helper
        - :class:`torch_concepts.annotations.Annotations`: Concept annotations
        - `TorchMetrics Documentation <https://lightning.ai/docs/torchmetrics>`_: 
          Available metrics and their parameters
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
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, split: str = 'train'):
        """Update metrics with predictions and targets for a given split.
        
        This method automatically routes predictions to the appropriate metrics based
        on concept types. For summary metrics, it aggregates all concepts of each type.
        For per-concept metrics, it extracts individual concept predictions.
        
        The preds tensor should be in the endogenous space (after applying the concept
        distributions' transformations), and the target tensor should contain the
        ground truth concept values.
        
        Args:
            preds (torch.Tensor): Model predictions in endogenous space. Shape depends
                on concept types:
                
                - Binary concepts: (batch_size, n_binary_concepts)
                - Categorical concepts: (batch_size, sum of cardinalities)
                - Mixed: (batch_size, n_binary + sum of cat cardinalities)
                
            target (torch.Tensor): Ground truth concept values. Shape (batch_size, n_concepts)
                where each column corresponds to a concept:
                
                - Binary concepts: float values in {0, 1}
                - Categorical concepts: integer class indices in {0, ..., cardinality-1}
                - Continuous concepts: float values (not yet supported)
                
            split (str, optional): Which data split to update. Must be one of:
                
                - 'train': Training split
                - 'val' or 'validation': Validation split
                - 'test': Test split
                
                Defaults to 'train'.
                
        Raises:
            ValueError: If split is not one of 'train', 'val', 'validation', or 'test'
            NotImplementedError: If continuous concepts are encountered
            
        Example:
            **Basic update**::
            
                # Binary concepts only
                predictions = torch.randn(32, 3)  # 3 binary concepts
                targets = torch.randint(0, 2, (32, 3))  # binary ground truth
                
                metrics.update(preds=predictions, target=targets, split='train')
                
            **Mixed concept types**::
                
                # 2 binary + 1 categorical (3 classes)
                # Endogenous space: 2 binary + 3 categorical = 5 dims
                predictions = torch.randn(32, 5)
                targets = torch.cat([
                    torch.randint(0, 2, (32, 2)),  # binary targets
                    torch.randint(0, 3, (32, 1))   # categorical target
                ], dim=1)
                
                metrics.update(preds=predictions, target=targets, split='train')
                
            **Validation split**::
                
                val_predictions = model(val_data)
                metrics.update(preds=val_predictions, target=val_targets, split='val')        Note:
            - This method accumulates metric state across multiple batches
            - Call :meth:`compute` to calculate final metric values
            - Call :meth:`reset` after computing to start fresh for next epoch
            - Each split maintains independent state
        """
        # Skip empty batches to avoid errors in underlying metric libraries
        if preds.shape[0] == 0:
            return
        
        metric_collection = self._get_collection(split)
        
        for key in metric_collection:
            # Update summary metrics
            if self.summary_metrics:
                if 'SUMMARY-binary_' in key and self.groups['binary_labels']:
                    binary_pred = preds[:, self.groups['binary_endogenous_idx']]
                    binary_target = target[:, self.groups['binary_idx']].float()
                    metric_collection[key].update(binary_pred, binary_target)
                    continue
                
                elif 'SUMMARY-categorical_' in key and self.groups['categorical_labels']:
                    # Pad and stack categorical endogenous
                    split_tuple = torch.split(
                        preds[:, self.groups['categorical_endogenous_idx']], 
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
                    cat_pred = torch.cat(padded_endogenous, dim=0)
                    cat_target = target[:, self.groups['categorical_idx']].T.reshape(-1).long()
                    metric_collection[key].update(cat_pred, cat_target)
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
                        preds[:, endogenous_idx], 
                        target[:, c_idx:c_idx+1].float()
                    )
                elif c_type == 'discrete' and card > 1:
                    metric_collection[key].update(
                        preds[:, endogenous_idx], 
                        target[:, c_idx].long()
                    )
                elif c_type == 'continuous':
                    metric_collection[key].update(
                        preds[:, endogenous_idx], 
                        target[:, c_idx:c_idx+1]
                    )
                else:
                    raise ValueError(f"ConceptMetrics.update(): Unknown concept \
                                     type '{c_type}' for concept '{concept_name}'.")
    
    def compute(self, split: str = 'train'):
        """Compute final metric values from accumulated state for a split.
        
        This method calculates the final metric values using all data accumulated
        through :meth:`update` calls since the last :meth:`reset`. It does not
        reset the metric state, allowing you to log results before resetting.
        
        Args:
            split (str, optional): Which data split to compute metrics for.
                Must be one of 'train', 'val', 'validation', or 'test'.
                Defaults to 'train'.
                
        Returns:
            dict: Dictionary mapping metric names (with split prefix) to computed
                values. Keys follow the format:
                
                - Summary metrics: '{split}/SUMMARY-{type}_{metric_name}'
                - Per-concept metrics: '{split}/{concept_name}_{metric_name}'
                
                Values are torch.Tensor objects containing the computed metric values.
                
        Raises:
            ValueError: If split is not one of the valid options
            
        Example:
            **Basic compute**::
            
                # After updating with training data
                train_results = metrics.compute('train')
                print(train_results)
                # {
                #     'train/SUMMARY-binary_accuracy': tensor(0.8500),
                #     'train/SUMMARY-binary_f1': tensor(0.8234),
                #     'train/concept1_accuracy': tensor(0.9000),
                #     'train/concept2_accuracy': tensor(0.8000)
                # }
                
            **Compute multiple splits**::
            
                train_metrics = metrics.compute('train')
                val_metrics = metrics.compute('val')
                
                # Log to wandb or tensorboard
                logger.log_metrics(train_metrics)
                logger.log_metrics(val_metrics)
                
            **Extract specific metrics**::
            
                results = metrics.compute('val')
                accuracy = results['val/SUMMARY-binary_accuracy'].item()
                print(f"Validation accuracy: {accuracy:.2%}")
                
        Note:
            - This method can be called multiple times without resetting
            - Always call :meth:`reset` after logging to start fresh for next epoch
            - Returned tensors are on the same device as the metric state
        """
        metric_collection = self._get_collection(split)
        return metric_collection.compute()
    
    def reset(self, split: Optional[str] = None):
        """Reset metric state for one or all splits.
        
        This method resets the accumulated metric state, clearing all data from
        previous :meth:`update` calls. Call this after computing and logging metrics
        to prepare for the next epoch.
        
        Args:
            split (Optional[str], optional): Which split to reset. Options:
                
                - 'train': Reset only training metrics
                - 'val' or 'validation': Reset only validation metrics
                - 'test': Reset only test metrics
                - None: Reset all splits simultaneously (default)
                
        Raises:
            ValueError: If split is not None and not a valid split name
            
        Example:
            **Reset single split**::
            
                # At end of training epoch
                train_metrics = metrics.compute('train')
                logger.log_metrics(train_metrics)
                metrics.reset('train')  # Reset only training
                
            **Reset all splits**::
            
                # At end of validation
                train_metrics = metrics.compute('train')
                val_metrics = metrics.compute('val')
                logger.log_metrics({**train_metrics, **val_metrics})
                metrics.reset()  # Reset both train and val
                
            **Typical training loop**::
            
                for epoch in range(num_epochs):
                    # Training
                    for batch in train_loader:
                        preds = model(batch)
                        metrics.update(preds, targets, split='train')
                    
                    # Validation
                    for batch in val_loader:
                        preds = model(batch)
                        metrics.update(preds, targets, split='val')
                    
                    # Compute and log
                    train_results = metrics.compute('train')
                    val_results = metrics.compute('val')
                    log_metrics({**train_results, **val_results})
                    
                    # Reset for next epoch
                    metrics.reset()  # Resets both train and val
                    
        Note:
            - Resetting is essential to avoid mixing data from different epochs
            - Each split can be reset independently
            - Resetting does not affect the metric configuration, only the state
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
