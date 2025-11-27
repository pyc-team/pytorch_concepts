"""PyTorch Lightning training engine for concept-based models.

This module provides the Predictor class, which orchestrates the training, 
validation, and testing of concept-based models. It handles:
- Loss computation with type-aware losses (binary/categorical/continuous)
- Metric tracking (summary and per-concept)
- Optimizer and scheduler configuration
- Batch preprocessing and transformations
- Concept interventions (experimental)
"""

from typing import Optional, Mapping, Union
from abc import abstractmethod

from torch import nn
import torch
from torchmetrics import MetricCollection
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import Optimizer, LRScheduler

from .....nn.modules.metrics import ConceptMetrics


class BaseLearner(pl.LightningModule):
    """
    Base training engine for concept-based models (PyTorch Lightning).

    Handles loss, metrics, optimizer, scheduler, batch validation, and logging.

    Args:
        loss (nn.Module, optional): Loss function.
        metrics (ConceptMetrics or dict, optional): Metrics for evaluation.
        optim_class (Optimizer, optional): Optimizer class.
        optim_kwargs (dict, optional): Optimizer arguments.
        scheduler_class (LRScheduler, optional): Scheduler class.
        scheduler_kwargs (dict, optional): Scheduler arguments.

    Example:
        >>> from torch_concepts.nn.modules.high.base.learner import BaseLearner
        >>> from torch_concepts.nn.modules.metrics import ConceptMetrics, GroupConfig
        >>> learner = BaseLearner(loss=None, metrics=None)
    """
    def __init__(self,
                loss: Optional[nn.Module] = None,
                metrics: Optional[Union[ConceptMetrics, Mapping[str, MetricCollection]]] = None,
                optim_class: Optional[Optimizer] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[LRScheduler] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                **kwargs
    ):        
        super(BaseLearner, self).__init__(**kwargs)

        # loss function
        self.loss = loss

        # optimizer and scheduler
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

        # metrics object
        self.metrics = metrics
        # Create pointers to individual collections for consistent interface
        # Both dict.get() and ConceptMetrics.get() return None if key doesn't exist
        if metrics is not None:
            if isinstance(metrics, dict):
                # Validate dict keys are correct
                assert all(key in ['train_metrics', 'val_metrics', 'test_metrics'] for key in metrics), (
                    "metrics dict keys must be 'train_metrics', 'val_metrics', and/or 'test_metrics'."
                )
            self.train_metrics = metrics.get('train_metrics')
            self.val_metrics = metrics.get('val_metrics')
            self.test_metrics = metrics.get('test_metrics')
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

    def __repr__(self):
        scheduler_name = self.scheduler_class.__name__ if self.scheduler_class else None
        return (f"{self.__class__.__name__}(n_concepts={self.n_concepts}, "
                f"optimizer={self.optim_class.__name__}, scheduler={scheduler_name})")

    def update_metrics(self, preds: torch.Tensor, target: torch.Tensor, step: str):
        """Update metrics with predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.
            step (str): Which split to update ('train', 'val', or 'test').
        """
        if self.metrics is None:
            return
            
        if isinstance(self.metrics, dict):
            # Update the appropriate MetricCollection directly
            collection = getattr(self, f"{step}_metrics", None)
            if collection is not None:
                collection.update(preds, target)
        elif isinstance(self.metrics, ConceptMetrics):
            # ConceptMetrics handles split internally
            self.metrics.update(preds, target, step)
        else:
            raise ValueError("Metrics must be either a ConceptMetrics object \
                             or a dict of MetricCollections.")

    def update_and_log_metrics(self, metrics_args: Mapping, step: str, batch_size: int):
        """Update metrics and log them.
        
        Args:
            metrics_args (Mapping): Dict with 'preds' and 'target' for metrics.
                This is the standard signature for torchmetrics Metrics.
            step (str): Which split to update ('train', 'val', or 'test').
            batch_size (int): Batch size for metric logging.
        """
        preds = metrics_args['preds']
        target = metrics_args['target']
        self.update_metrics(preds, target, step)
        
        # Get the collection to log
        collection = getattr(self, f"{step}_metrics", None)
        if collection is not None:
            self.log_metrics(collection, batch_size=batch_size)

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

    def _check_batch(self, batch):
        """Validate batch structure and required keys.
        
        Args:
            batch (dict): Batch dictionary from dataloader.
        Raises:
            KeyError: If required keys 'inputs' or 'concepts' are missing from batch
        """
        # Validate batch structure
        if not isinstance(batch, dict):
            raise TypeError(
                f"Expected batch to be a dict, but got {type(batch).__name__}. "
                f"Ensure your dataset returns batches as dictionaries with 'inputs' and 'concepts' keys."
            )
        
        required_keys = ['inputs', 'concepts']
        # TODO: add option to train an unsupervised concept-based model
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise KeyError(
                f"Batch is missing required keys: {missing_keys}. "
                f"Found keys: {list(batch.keys())}. "
                f"Ensure your dataset returns batches with 'inputs' and 'concepts' keys."
            )

    def unpack_batch(self, batch):
        """Extract inputs, concepts, and transforms from batch dict.
        can be overridden by model-specific preprocessing.
        
        Args:
            batch (dict): Batch with 'inputs', 'concepts', and optional 'transform'.
            
        Returns:
            Tuple: (inputs, concepts, transforms) after model-specific preprocessing.
        """
        self._check_batch(batch)
        inputs = batch['inputs']
        concepts = batch['concepts']
        transforms = batch.get('transforms', {})
        return inputs, concepts, transforms

    # TODO: implement input preprocessing with transforms from batch
    # @staticmethod
    # def maybe_apply_preprocessing(preprocess: bool,
    #                               inputs: Mapping,
    #                               transform: Mapping) -> torch.Tensor:
    #     # apply batch preprocessing
    #     if preprocess:
    #         for key, transf in transform.items():
    #             if key in inputs:
    #                 inputs[key] = transf.transform(inputs[key])
    #     return inputs

    # TODO: implement concepts rescaling with transforms from batch
    # @staticmethod
    # def maybe_apply_postprocessing(postprocess: bool,
    #                                forward_out: Union[torch.Tensor, Mapping],
    #                                transform: Mapping) -> torch.Tensor:
    #     raise NotImplementedError("Postprocessing is not implemented yet.")
    #     # apply batch postprocess
    #     if postprocess:
    #         case isinstance(forward_out, Mapping):
    #             ....

    #         case isinstance(forward_out, torch.Tensor):
    #             only continuous concepts...
    #             transf = transform.get('c')
    #             if transf is not None:
    #                 out = transf.inverse_transform(forward_out)
    #     return out

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
            Union[Optimizer, dict, None]: Returns optimizer directly, or dict with 
                'optimizer' and optionally 'lr_scheduler' and 'monitor' keys,
                or None if no optimizer is configured.
        """
        # No optimizer configured
        if self.optim_class is None:
            return None
        
        # Initialize optimizer with proper kwargs handling
        optim_kwargs = self.optim_kwargs if self.optim_kwargs is not None else {}
        optimizer = self.optim_class(self.parameters(), **optim_kwargs)
        
        # No scheduler configured - return optimizer directly
        if self.scheduler_class is None:
            return {"optimizer": optimizer}
        
        # Scheduler configured - build configuration dict
        # Make a copy to avoid modifying original kwargs
        scheduler_kwargs = self.scheduler_kwargs.copy() if self.scheduler_kwargs is not None else {}
        monitor_metric = scheduler_kwargs.pop("monitor", None)
        
        scheduler = self.scheduler_class(optimizer, **scheduler_kwargs)
        
        cfg = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
        
        # Add monitor metric if specified (required for ReduceLROnPlateau)
        if monitor_metric is not None:
            cfg["monitor"] = monitor_metric
        
        return cfg
 