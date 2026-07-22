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
        scalers (ScalerModule, optional): Scalers fitted on the training split
            (see :attr:`ConceptDataModule.fitted_scalers
            <torch_concepts.data.base.datamodule.ConceptDataModule.fitted_scalers>`).
            When given, everything the model touches lives in **scaled** space
            while metrics are reported in the **original** scale — see
            :meth:`shared_step`. When None, no scaling happens anywhere.

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
                scalers: Optional[nn.Module] = None,
                **kwargs
    ):
        super(BaseLearner, self).__init__(**kwargs)

        # Fitted data scalers (a ScalerModule). Assigned as a submodule so its
        # statistics follow the model across devices and into checkpoints.
        self.scalers = scalers

        # loss function. Only a TypeAwareLoss (e.g. ConceptLoss), which consumes
        # the whole ModelOutput, is supported.
        self.loss = loss
        self._loss_takes_model_output = isinstance(loss, TypeAwareLoss)

        # optimizer and scheduler
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

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
        transforms = batch.get('transforms', {})
        return inputs, concepts, transforms

    # --------------------------- Scaling ---------------------------

    @cached_property
    def _continuous_concepts(self):
        """``(labels, column_indices, is_every_concept)`` for the continuous
        concepts, computed once.

        The concept axis is fixed after construction, so the lookup that
        :meth:`scale_concepts` needs on every batch is derived a single time. 
        Labels and indices are in concept-space: one column per concept. 
        ``is_every_concept`` says the continuous columns are *all* the columns, 
        which lets :meth:`scale_concepts` transform the tensor whole instead of paying 
        a clone plus a gather/scatter — the common case for a regression dataset.
        """
        axis = self.concept_annotations
        labels = [name for name in axis.labels if axis.concept(name).type == 'continuous']
        indices = [axis.get_index(name) for name in labels]
        return labels, indices, len(labels) == len(axis.labels)

    def scale_inputs(self, inputs: Mapping) -> Mapping:
        """Scale the raw input the model consumes.

        Returns *inputs* unchanged when no input scaler was fitted.
        """
        if self.scalers is None or not self.scalers.has_input:
            return inputs
        return {**inputs, 'x': self.scalers.transform_input(inputs['x'])}

    def scale_concepts(self, c):
        """Scale the continuous columns of a ``(batch, n_concepts)`` ground truth.

        Binary and categorical columns are class labels and pass through
        untouched. The result keeps the concept-space annotation, so it can be
        fed to :meth:`build_query` and :meth:`prepare_target` exactly like the
        raw batch tensor. Returns *c* unchanged when no concept scaler was fitted.
        """
        if c is None or self.scalers is None or not self.scalers.has_concepts:
            return c
        labels, indices, is_every_concept = self._continuous_concepts
        if not labels:
            return c
        annotation = getattr(c, 'annotation', None)
        if annotation is None:
            annotation = self.concept_annotations.to_concept_space()
        data = c.tensor if isinstance(c, AnnotatedTensor) else c
        if is_every_concept:
            scaled = self.scalers.transform_concepts(data, labels).to(data.dtype)
        else:
            scaled = data.clone()
            scaled[..., indices] = self.scalers.transform_concepts(
                data[..., indices], labels
            ).to(scaled.dtype)
        return AnnotatedTensor(scaled, annotation, getattr(c, 'axis', -1))

    def unscale_output(self, out: ModelOutput) -> ModelOutput:
        """Map a model output's continuous point estimates back to the original scale.

        Only ``loc`` and ``value`` are inverted — the quantities a continuous
        concept is *scored* on (see ``CONTINUOUS_QUANTITIES``), and the only ones
        :class:`~torch_concepts.nn.ConceptMetrics` reads. ``scale`` /
        ``scale_tril`` / ``samples`` stay in scaled space: they are dispersion and
        draws, whose inverse is not the point estimate's, and the loss that
        consumes them runs in scaled space anyway.

        Columns without a fitted scaler pass through untouched, so a query that
        also asks for a non-concept variable — ``latent`` is a ``Delta`` and
        reports under ``value``, like a continuous concept would — is inverted on
        its concept columns only rather than rejected.

        The returned :class:`ModelOutput` is a shallow copy — every other field is
        carried over by reference, and *out* itself is left untouched.
        """
        if self.scalers is None or not self.scalers.has_concepts:
            return out
        scaled_names = set(self.scalers.concept_names)
        params = ParamsDict(out.params)
        for quantity in CONTINUOUS_QUANTITIES:
            tensor = out.params.get(quantity)
            if tensor is None:
                continue
            annotation = tensor.annotation
            labels = [name for name in annotation.labels if name in scaled_names]
            if not labels:
                continue
            # Quantity tensors are (*leading, width) with the annotated axis last
            # (the InferenceOutput contract), so the columns index with `...`.
            data = tensor.tensor
            if len(labels) == data.shape[-1]:
                # Every column is a scaled concept: invert the tensor whole and
                # skip the clone plus gather/scatter.
                restored = self.scalers.inverse_concepts(data, labels).to(data.dtype)
            else:
                indices = annotation.get_slice(labels)
                restored = data.clone()
                restored[..., indices] = self.scalers.inverse_concepts(
                    data[..., indices], labels
                ).to(restored.dtype)
            params[quantity] = AnnotatedTensor(restored, annotation, tensor.axis)
        return ModelOutput(
            params=params,
            guide_params=out.guide_params,
            samples=out.samples,
            probabilities=out.probabilities,
            target=out.target,
            extra=out.extra,
        )

    @cached_property
    def concept_variables(self):
        """List of concept variable names (plate names or individual concepts)."""
        return [var.name for var in self.pgm.variables.values() if var.variable_type == 'concept']

    def default_query(self, c):
        """Default query for a training/eval step: observe **every** concept,
        teacher-forced to its ground-truth value (via :meth:`build_query`).

        This is the full-observation query the standard step uses. Override in a
        learner that should observe only a subset of concepts (or leave them
        latent) — e.g. a task-only learner.
        """
        return self.build_query(c)

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
        inputs, concepts, _ = self.unpack_batch(batch)

        # TODO: needs to extend to arbitrary leading dims (e.g., for text)
        batch_size = batch['inputs']['x'].size(0)
        c = concepts.get('c', None)

        # The query is built from the *full* concept tensor even for models that
        # predict a subset: BlackBoxTaskOnly.build_query indexes it with offsets
        # into the whole concept axis.
        c_scaled = self.scale_concepts(c)

        # --- Model forward ---
        # Defaults: observe all concepts, pass the input as evidence.
        query = self.default_query(c_scaled)
        out = self.forward(
            query=query,
            evidence=self.default_evidence(self.scale_inputs(inputs)),
        )

        # prepare_target returns a concept-space annotated target (and slices it
        # for models like BlackBoxTaskOnly that predict a subset of concepts).
        # It is applied to each scaling of the ground truth separately, since that
        # slicing has to happen on both.
        target = self.prepare_target(c_scaled)

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
        self.update_and_log_metrics(
            self.unscale_output(out), self.prepare_target(c), step, batch_size
        )
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
 