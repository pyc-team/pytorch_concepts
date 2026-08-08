"""PyTorch Lightning training engine for concept-based models.

This module provides the BaseLearner class, which handle training, validation, and testing 
of concept-based models. Specifically, it allows to use utilities of PyTorch Lightning 
for updating model parameters, computing and logging metrics, and configuring optimizers and schedulers.
The training and evaluation logic is instead delegated to the 'inference' object of the model.

It handles:
- Loss computation with type-aware losses (binary/categorical/continuous)
- Metric tracking (summary and per-concept)
- Optimizer and scheduler configuration
- Batch preprocessing and transformations
- Model evaluation
"""

from typing import Optional, Mapping
from functools import cached_property

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import Optimizer, LRScheduler

from .....tensor import AnnotatedTensor
from ...metrics import ConceptMetrics
from ...loss import TypeAwareLoss
from ...outputs import CONTINUOUS_QUANTITIES, ModelOutput, ParamsDict


class BaseLearner(pl.LightningModule):
    """
    Base training engine for concept-based models (PyTorch Lightning).

    Handles loss, metrics, optimizer, scheduler, batch validation, and logging.

    Args:
        loss (nn.Module, optional): Loss function for training.  Use per-type composition via ``ConceptLoss``
            to combine multiple terms (see ``binary``, ``binary_weights``,
            etc.).
        metrics (ConceptMetrics, optional): Metrics for evaluation.
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
                metrics: Optional[ConceptMetrics] = None,
                optim_class: Optional[Optimizer] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[LRScheduler] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                scale_concepts: bool = True,
                **kwargs
    ):
        super(BaseLearner, self).__init__(**kwargs)

        # loss function. Only a TypeAwareLoss (e.g. ConceptLoss), which consumes
        # the whole ModelOutput, is supported.
        self.loss = loss
        self._loss_takes_model_output = isinstance(loss, TypeAwareLoss)

        # optimizer and scheduler
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

        # whether to compute the loss in the scaled space for the continuous concepts
        self.scale_concepts = scale_concepts

        # Create pointers to train, val and test collections
        if isinstance(metrics, ConceptMetrics) and metrics.collection:
            self.setup_metrics(metrics)
        elif metrics is not None:
            assert isinstance(metrics, ConceptMetrics), (
                f"metrics must be a ConceptMetrics instance, got {type(metrics)}"
            )
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

    def __repr__(self):
        scheduler_name = self.scheduler_class.__name__ if self.scheduler_class else None
        return (f"{self.__class__.__name__}(n_concepts={self.n_concepts}, "
                f"optimizer={self.optim_class.__name__}, scheduler={scheduler_name})")
    
    def setup_metrics(self, metrics: ConceptMetrics):
        self.train_metrics = metrics.clone(prefix="train")
        self.val_metrics = metrics.clone(prefix="val")
        self.test_metrics = metrics.clone(prefix="test") 

    def update_and_log_metrics(self, out: ModelOutput, target, step: str, batch_size: int):
        """Update metrics and log them.

        Args:
            out (ModelOutput): Model output containing the predictions.
            target: Concept-space ground truth.
            step (str): Which split to update ('train', 'val', or 'test').
            batch_size (int): Batch size for metric logging.
        """
        self.update_metrics(out, target, step)

        # Get the collection to log
        collection = getattr(self, f"{step}_metrics", None)
        if collection is not None:
            self.log_metrics(collection, batch_size=batch_size)

    def update_metrics(self, out: ModelOutput, target, step: str):
        """Update metrics with model output and target.

        Args:
            out (ModelOutput): Model output containing the predictions.
            target: Concept-space ground truth.
            step (str): Which split to update ('train', 'val', or 'test').
        """
        collection = getattr(self, f"{step}_metrics", None)
        if collection is not None:
            collection.update(out, target)
        
    def log_metrics(self, metrics, **kwargs):
        """Log metrics to logger (W&B) at epoch end.
        
        Args:
            metrics: MetricCollection, ConceptMetrics, or dict of metrics.
            **kwargs: Additional arguments passed to self.log_dict.
        """
        if isinstance(metrics, ConceptMetrics):
            for coll in metrics.collection.values():
                self.log_dict(
                    coll, on_step=False, on_epoch=True,
                    logger=True, prog_bar=False, **kwargs
                )
        else:
            self.log_dict(
                metrics, on_step=False, on_epoch=True,
                logger=True, prog_bar=False, **kwargs
            )

    def log_loss(self, name, loss, **kwargs):
        """Log loss to logger and progress bar at epoch end.
        
        Args:
            name (str): Loss name prefix (e.g., 'train', 'val', 'test').
            loss (torch.Tensor): Loss value to log.
            **kwargs: Additional arguments passed to self.log.
        """
        self.log(
            name + "_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            **kwargs
        )

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
        transforms = batch.get('scalers', {})
        return inputs, concepts, transforms

    def maybe_scale_inputs(self, inputs, transforms):
        """Scale precomputed/tabular inputs with the batch's 'input' scaler, if
        one was shipped. A no-op otherwise (e.g. raw images).

        Args:
            inputs: Input dict (``{'x': ...}`` plus any extra keys).
            transforms: The batch's fitted scalers, keyed by 'input'/'concepts'.
        Returns:
            The input dict, with 'x' scaled when an 'input' scaler is present.
        """
        scaler = transforms.get('input')
        return {**inputs, 'x': scaler.transform(inputs['x'])} if scaler is not None else inputs

    def maybe_scale_concepts(self, concepts, transforms):
        """Scale ground-truth continuous concepts into the model's scaled space
        (used for both the teacher-forced query and the loss target); binary and
        categorical concepts pass through unchanged.

        A no-op when :attr:`scale_concepts` is off, no 'concepts' scaler was
        shipped, or the dataset has no continuous concepts.

        Args:
            concepts: Concept dict (``{'c': AnnotatedTensor}``).
            transforms: The batch's fitted scalers.
        Returns:
            dict: ``{'c': ...}``, with continuous columns standardized, or the
            concept tensor unchanged.
        """
        # raise error if multiple keys in 'concepts'
        if isinstance(concepts, dict) and len(concepts) > 1:
            raise NotImplementedError(
                f"Expected a single concept AnnotatedTensor, but got multiple keys: {list(concepts.keys())}. "
                f"review this current implementation when using multiple annotators. "
            )
        c = concepts.get('c')
        scaler = transforms.get('concepts') if self.scale_concepts else None
        if scaler is not None and c is not None:
            labels = c.annotation.labels_by_type.get('continuous')
            if labels:
                scaled = AnnotatedTensor(c.tensor.clone(), c.annotation, c.axis)
                scaled[labels] = scaler.transform(c[labels].tensor)
                c = scaled
        return {'c': c}

    def unscale_output(self, out, transforms):
        """Inverse-transform continuous predictions ('loc'/'value') back to
        natural units, so metrics are always reported on the original data scale.
        A no-op when :attr:`scale_concepts` is off or no 'concepts'
        scaler was shipped.

        Args:
            out: Model output whose continuous quantities are in scaled space.
            transforms: The batch's fitted scalers.
        Returns:
            The output with continuous quantities restored to the original scale.
        """
        scaler = transforms.get('concepts') if self.scale_concepts else None
        if scaler is None:
            return out
        for quantity in CONTINUOUS_QUANTITIES:  # ('loc', 'value')
            pred = out.params.get(quantity)
            if pred is None:
                continue
            setattr(out, quantity, AnnotatedTensor(
                scaler.inverse_transform(pred.tensor), pred.annotation, pred.axis
            ))
        return out

    @cached_property
    def concept_variables(self):
        """List of concept variable names (plate names or individual concepts)."""
        return [var.name for var in self.pgm.variables.values() if var.variable_type == 'concept']

    def default_query(self, c):
        """Default query for a training/eval step: observe **every** concept,
        teacher-forced to its ground-truth value (via :meth:`fully_observed_query`).

        This is the full-observation query the standard step uses. Override in a
        learner that should observe only a subset of concepts (or leave them
        latent) — e.g. a task-only learner.
        """
        return self.fully_observed_query(c)

    def default_evidence(self, inputs):
        """Default evidence for a training/eval step: the raw input only
        (``{"input": inputs["x"]}``).

        Override to supply additional observed (non-concept) variables.
        """
        return {"input": inputs["x"]}

    def shared_step(self, batch, step):
        """Shared logic for train/val/test steps.

        With ``scalers`` configured, the model runs entirely in **scaled** space —
        its evidence, the teacher-forced concept values in its query and its loss
        target are all scaled — while metrics are updated in the **original**
        data scale. The logged ``*_loss`` is therefore in scaled units; the logged
        metrics are not. Without scalers every step below is an identity and the
        behaviour is unchanged.

        Parameters
        ----------
        batch : dict
            Batch dictionary with 'inputs' and 'concepts' keys.
        step : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        inputs, concepts, transforms = self.unpack_batch(batch)

        # TODO: needs to extend to arbitrary leading dims (e.g., for text)
        batch_size = batch['inputs']['x'].size(0)

        inputs = self.maybe_scale_inputs(inputs, transforms)
        c_loss = self.maybe_scale_concepts(concepts, transforms).get('c', None)

        # --- Model forward (scaled space) ---
        # Defaults: observe all concepts, pass the input as evidence.
        query = self.default_query(c_loss)
        evidence = self.default_evidence(inputs)
        out = self.forward(query=query, evidence=evidence)

        # Publish the observed values so a loss term can score them. A generative
        # term (e.g. ReconstructionLoss) needs the *observed* variable, which is
        # in the evidence rather than in the concept target.
        out.extra = {**(out.extra or {}), "evidence": evidence}

        target = self.prepare_target(c_loss)

        # --- Compute loss (scaled space) ---
        loss = None
        if self.loss is not None:
            if not self._loss_takes_model_output:
                raise NotImplementedError(
                    "Only a TypeAwareLoss (e.g. ConceptLoss) is supported; a plain "
                    "loss(input, target) is not."
                )
            loss = self.loss(out, target)
            self.log_loss(step, loss, batch_size=batch_size)

        # --- Update and log metrics (original scale) ---
        out = self.unscale_output(out, transforms)
        target = self.prepare_target(concepts.get('c', None))
        self.update_and_log_metrics(out, target, step, batch_size)
        return loss

    def training_step(self, batch):
        """Training step called by PyTorch Lightning.
        
        Args:
            batch (dict): Training batch.
            
        Returns:
            torch.Tensor: Training loss.
        """
        # TODO: train interventions using the context manager 'with ...'
        loss = self.shared_step(batch, step='train')
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Advance the relaxation temperature once per optimiser step.

        *Both* engines, on the same schedule, so evaluation reads the temperature
        training reached instead of sitting at the initial one — testing a
        relaxation the model never trained at.

        After the backward pass, not at the end of ``training_step``: the
        temperature is part of the forward graph and is updated in place, so
        mutating it earlier would invalidate the graph autograd is about to walk.
        """
        engines = {id(e): e for e in (self.train_inference, self.eval_inference) if e}
        for engine in engines.values():
            engine.temperature_step()

    def validation_step(self, batch):
        """Validation step called by PyTorch Lightning.
        
        Args:
            batch (dict): Validation batch.
            
        Returns:
            torch.Tensor: Validation loss.
        """
        loss = self.shared_step(batch, step='val')
        return loss
    
    def test_step(self, batch):
        """Test step called by PyTorch Lightning.
        
        Args:
            batch (dict): Test batch.
            
        Returns:
            torch.Tensor: Test loss.
        """
        loss = self.shared_step(batch, step='test')
        
        # TODO: test-time interventions
        # self.test_intervention(batch)
        # if 'Qualified' in self.c_names:
        #     self.test_intervention_fairness(batch)

        return loss

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
 