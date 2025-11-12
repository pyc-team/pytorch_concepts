from typing import Optional, Mapping, Type, Tuple, Callable, Union
import warnings

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from torch_concepts import AxisAnnotation
from torch_concepts.nn import BaseInference

from ..utils import instantiate_from_string


class Predictor(pl.LightningModule):    
    def __init__(self,
                model: nn.Module,
                train_inference: BaseInference,
                loss: Mapping,
                metrics: Mapping,
                preprocess_inputs: bool = False,
                scale_concepts: bool = False,
                enable_summary_metrics: bool = True,
                enable_perconcept_metrics: bool = False,
                *,
                optim_class: Type,
                optim_kwargs: Mapping,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                train_interv_prob: Optional[float] = 0.,
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                ):
        
        super(Predictor, self).__init__()
 
        # instantiate model
        self.model = model
        
        # set training inference
        # FIXME: fix naming convention for models. model
        # is both the wrapper and the internal model 
        # also fix class names
        self.train_inference_engine = train_inference(self.model.pgm)

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

        # interventions for regularization purposes
        self.train_interv_prob = train_interv_prob

        # concept info
        self.concept_annotations = self.model.annotations.get_axis_annotation(1)
        self.concept_names = self.concept_annotations.labels
        self.n_concepts = len(self.concept_names)

        # Pre-compute concept grouping for efficient computation
        self._setup_concept_groups()

        # Setup and instantiate loss functions
        self._setup_losses(loss)

        # Setup and instantiate metrics
        self._setup_metrics(metrics)

    def __repr__(self):
        return "{}(model={}, n_concepts={}, train_interv_prob={}, " \
               "test_interv_policy={}, optimizer={}, scheduler={})" \
            .format(self.__class__.__name__,
                    self.model.__class__.__name__,
                    self.optim_class.__name__,
                    self.scheduler_class.__name__ if self.scheduler_class else None)

    def _setup_concept_groups(self):
        """Pre-compute concept information for efficient computation."""
        metadata = self.concept_annotations.metadata
        cardinalities = self.concept_annotations.cardinalities
        
        # Store per-concept info
        self.tasks = [metadata[name]['task'] for name in self.concept_names]
        self.cardinalities = cardinalities
        self.is_nested = self.concept_annotations.is_nested

    def _check_collection(self, 
                          annotations: AxisAnnotation, 
                          collection: Mapping,
                          collection_name: str):
        """
        Validate collections (typically metrics and losses) against concept annotations.
        Discards unused collection items and performs sanity checks.
        """
        assert collection_name in ['loss', 'metrics'], "collection_name must be either 'loss' or 'metrics'"

        # Extract annotation properties
        metadata = annotations.metadata
        cardinalities = annotations.cardinalities
        tasks = [c_meta['task'] for _, c_meta in metadata.items()]
        
        # Categorize concepts by task and cardinality
        is_binary = [t == 'classification' and card == 1 for t, card in zip(tasks, cardinalities)]
        is_categorical = [t == 'classification' and card > 1 for t, card in zip(tasks, cardinalities)]
        is_regression = [t == 'regression' for t in tasks]
        
        has_binary = any(is_binary)
        has_categorical = any(is_categorical)
        has_regression = any(is_regression)
        all_same_task = all(t == tasks[0] for t in tasks)
        
        # Determine required collection items
        needs_binary = has_binary
        needs_categorical = has_categorical
        needs_regression = has_regression
        
        # Helper to get collection item or None
        def get_item(path):
            try:
                result = collection
                for key in path:
                    result = result[key]
                return result
            except (KeyError, TypeError):
                return None
        
        # Extract items from collection
        binary = get_item(['classification', 'binary'])
        categorical = get_item(['classification', 'categorical'])
        regression = get_item(['regression'])
        
        # Validation rules
        errors = []
        
        # Check nested/dense compatibility
        if all(is_binary):
            if annotations.is_nested:
                errors.append("Annotations for all-binary concepts should NOT be nested.")
            if not all_same_task:
                errors.append("Annotations for all-binary concepts should share the same task.")
        
        elif all(is_categorical):
            if not annotations.is_nested:
                errors.append("Annotations for all-categorical concepts should be nested.")
            if not all_same_task:
                errors.append("Annotations for all-categorical concepts should share the same task.")
        
        elif all(is_regression):
            if annotations.is_nested:
                errors.append("Annotations for all-regression concepts should NOT be nested.")
        
        elif has_binary or has_categorical:
            if not annotations.is_nested:
                errors.append("Annotations for mixed concepts should be nested.")
        
        # Check required items are present
        if needs_binary and binary is None:
            errors.append(f"{collection_name} missing 'classification.binary' for binary concepts.")
        if needs_categorical and categorical is None:
            errors.append(f"{collection_name} missing 'classification.categorical' for categorical concepts.")
        if needs_regression and regression is None:
            errors.append(f"{collection_name} missing 'regression' for regression concepts.")
        
        if errors:
            raise ValueError(f"{collection_name} validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # Warnings for unused items
        if not needs_binary and binary is not None:
            warnings.warn(f"Binary {collection_name} will be ignored (no binary concepts).")
        if not needs_categorical and categorical is not None:
            warnings.warn(f"Categorical {collection_name} will be ignored (no categorical concepts).")
        if not needs_regression and regression is not None:
            warnings.warn(f"Regression {collection_name} will be ignored (no regression concepts).")
        
        # Log configuration
        concept_types = []
        if has_binary and has_categorical:
            concept_types.append("mixed classification")
        elif has_binary:
            concept_types.append("all binary")
        elif has_categorical:
            concept_types.append("all categorical")
        
        if has_regression:
            concept_types.append("regression" if not (has_binary or has_categorical) else "with regression")
        
        print(f"{collection_name} configuration validated ({', '.join(concept_types)}):")
        print(f"  Binary (card=1): {binary if needs_binary else 'unused'}")
        print(f"  Categorical (card>1): {categorical if needs_categorical else 'unused'}")
        print(f"  Regression: {regression if needs_regression else 'unused'}")
        
        # Return only needed items (others set to None)
        return (binary if needs_binary else None,
                categorical if needs_categorical else None,
                regression if needs_regression else None)
    
    def _setup_losses(self, loss_config: Mapping):
        """Setup and instantiate loss functions."""
        # Validate and extract needed losses
        binary_cfg, categorical_cfg, regression_cfg = self._check_collection(
            self.concept_annotations, loss_config, 'loss'
        )
        
        # Instantiate loss functions
        self.binary_loss_fn = instantiate_from_string(binary_cfg['path'], **binary_cfg.get('kwargs', {})) if binary_cfg else None
        self.categorical_loss_fn = instantiate_from_string(categorical_cfg['path'], **categorical_cfg.get('kwargs', {})) if categorical_cfg else None
        self.regression_loss_fn = instantiate_from_string(regression_cfg['path'], **regression_cfg.get('kwargs', {})) if regression_cfg else None

    @staticmethod
    def _check_metric(metric):
        """Clone and reset a metric for use in collections."""
        metric = metric.clone()
        metric.reset()
        return metric

    def _setup_metrics(self, metrics_config: Mapping):
        """Setup and instantiate metrics with summary and/or per-concept options."""
        if metrics_config is None:
            metrics_config = {}
        
        # Validate and extract needed metrics
        binary_metrics_cfg, categorical_metrics_cfg, regression_metrics_cfg = self._check_collection(
            self.concept_annotations, metrics_config, 'metrics'
        )
        
        # Identify which concepts belong to which type
        self.binary_concept_ids = [i for i, (t, c) in enumerate(zip(self.tasks, self.cardinalities)) 
                                   if t == 'classification' and c == 1]
        self.categorical_concept_ids = [i for i, (t, c) in enumerate(zip(self.tasks, self.cardinalities)) 
                                        if t == 'classification' and c > 1]
        self.regression_concept_ids = [i for i, t in enumerate(self.tasks) if t == 'regression']
        
        # Initialize metric storage
        self.summary_metrics = {}
        self.perconcept_metrics = []
        
        # Setup summary metrics (one per type group)
        if self.enable_summary_metrics:
            if binary_metrics_cfg and self.binary_concept_ids:
                self.summary_metrics['binary'] = self._instantiate_metric_dict(binary_metrics_cfg)
            
            if categorical_metrics_cfg and self.categorical_concept_ids:
                # For categorical, we'll average over individual concept metrics
                self.summary_metrics['categorical'] = self._instantiate_metric_dict(
                    categorical_metrics_cfg, 
                    num_classes=max([self.cardinalities[i] for i in self.categorical_concept_ids])
                )
            
            if regression_metrics_cfg and self.regression_concept_ids:
                self.summary_metrics['regression'] = self._instantiate_metric_dict(regression_metrics_cfg)
        
        # Setup per-concept metrics (one per concept)
        if self.enable_perconcept_metrics:
            for c_id, concept_name in enumerate(self.concept_names):
                task = self.tasks[c_id]
                card = self.cardinalities[c_id]
                
                # Select the appropriate metrics config for this concept
                if task == 'classification' and card == 1:
                    metrics_cfg = binary_metrics_cfg
                elif task == 'classification' and card > 1:
                    metrics_cfg = categorical_metrics_cfg
                elif task == 'regression':
                    metrics_cfg = regression_metrics_cfg
                else:
                    metrics_cfg = None
                
                # Instantiate metrics for this concept
                concept_metric_dict = {}
                if metrics_cfg is not None:
                    for metric_name, metric_dict in metrics_cfg.items():
                        kwargs = metric_dict.get('kwargs', {})
                        if task == 'classification' and card > 1:
                            kwargs['num_classes'] = card
                        concept_metric_dict[metric_name] = instantiate_from_string(metric_dict['path'], **kwargs)
                
                self.perconcept_metrics.append(concept_metric_dict)
        else:
            # Empty dicts for all concepts if per-concept metrics disabled
            self.perconcept_metrics = [{} for _ in range(self.n_concepts)]
        
        # Create metric collections for train/val/test
        self._create_metric_collections()
    
    def _instantiate_metric_dict(self, metrics_cfg: Mapping, num_classes: int = None) -> dict:
        """Instantiate a dictionary of metrics from config."""
        if not isinstance(metrics_cfg, dict):
            return {}
        
        metrics = {}
        for metric_name, metric_path in metrics_cfg.items():
            kwargs = metric_path.get('kwargs', {})
            if num_classes is not None:
                kwargs['num_classes'] = num_classes
            metrics[metric_name] = instantiate_from_string(metric_path['path'], **kwargs)
        return metrics

    def _create_metric_collections(self):
        """Create MetricCollection for train/val/test from summary and per-concept metrics."""
        all_metrics = {}
        
        # Add summary metrics
        if self.enable_summary_metrics:
            for group_name, metric_dict in self.summary_metrics.items():
                for metric_name, metric in metric_dict.items():
                    key = f"{group_name}_{metric_name}"
                    all_metrics[key] = metric
        
        # Add per-concept metrics
        if self.enable_perconcept_metrics:
            for c_id, concept_name in enumerate(self.concept_names):
                for metric_name, metric in self.perconcept_metrics[c_id].items():
                    key = f"{concept_name}_{metric_name}"
                    all_metrics[key] = metric
        
        if not all_metrics:
            all_metrics = {}
        
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
                         regression_fn: Optional[Callable],
                         is_metric: bool = False) -> Union[torch.Tensor, None]:
        """
        Apply loss or metric functions by looping over concepts.
        
        Args:
            c_hat: Predicted concepts
            c_true: Ground truth concepts
            binary_fn: Function to apply to binary concepts
            categorical_fn: Function to apply to categorical concepts
            regression_fn: Function to apply to regression concepts
            is_metric: If True, updates metrics; if False, computes loss
            
        Returns:
            For losses: scalar tensor
            For metrics: None (metrics are updated in-place)
        """
        if not self.is_nested:
            # Dense format: apply to all concepts at once
            task = self.tasks[0]  # All tasks are the same in dense format
            card = self.cardinalities[0]
            
            if task == 'classification' and card == 1 and binary_fn:
                result = binary_fn(c_hat, c_true.float())
            elif task == 'regression' and regression_fn:
                result = regression_fn(c_hat, c_true)
            else:
                result = None
            
            return None if is_metric else result
        else:
            # Nested format: loop over concepts with different sizes
            concept_tensors = torch.split(c_hat, self.cardinalities, dim=1)
            total_loss = 0.0 if not is_metric else None
            
            for c_id, concept_tensor in enumerate(concept_tensors):
                task = self.tasks[c_id]
                card = self.cardinalities[c_id]
                c_true_i = c_true[:, c_id]
                
                if task == 'classification' and card == 1 and binary_fn:
                    result = binary_fn(concept_tensor, c_true_i.float().unsqueeze(1))
                    if not is_metric:
                        total_loss += result
                elif task == 'classification' and card > 1 and categorical_fn:
                    result = categorical_fn(concept_tensor, c_true_i.long())
                    if not is_metric:
                        total_loss += result
                elif task == 'regression' and regression_fn:
                    result = regression_fn(concept_tensor, c_true_i.unsqueeze(1))
                    if not is_metric:
                        total_loss += result
            
            return total_loss

    def _compute_loss(self, c_hat: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using pre-configured loss functions.
        
        Args:
            c_hat: Predicted concepts (logits or probabilities)
            c_true: Ground truth concepts
            
        Returns:
            Scalar loss value
        """
        return self._apply_fn_by_type(
            c_hat, c_true,
            self.binary_loss_fn,
            self.categorical_loss_fn,
            self.regression_loss_fn,
            is_metric=False
        )
    
    def _update_metrics(self, c_hat: torch.Tensor, c_true: torch.Tensor, 
                       metric_collection: MetricCollection):
        """
        Update both summary and per-concept metrics.
        
        Args:
            c_hat: Predicted concepts
            c_true: Ground truth concepts
            metric_collection: MetricCollection to update
        """
        # Update summary metrics (one per type group)
        if self.enable_summary_metrics:
            if not self.is_nested:
                # Dense format: apply to all concepts at once
                if self.binary_concept_ids:
                    for metric_name in self.summary_metrics.get('binary', {}).keys():
                        key = f"binary_{metric_name}"
                        if key in metric_collection:
                            metric_collection[key](c_hat, c_true.float())
                
                if self.regression_concept_ids:
                    for metric_name in self.summary_metrics.get('regression', {}).keys():
                        key = f"regression_{metric_name}"
                        if key in metric_collection:
                            metric_collection[key](c_hat, c_true)
            else:
                # Nested format: handle each type group
                concept_tensors = torch.split(c_hat, self.cardinalities, dim=1)
                
                # Binary group
                if self.binary_concept_ids:
                    binary_hats = [concept_tensors[i] for i in self.binary_concept_ids]
                    binary_trues = [c_true[:, i].float().unsqueeze(1) for i in self.binary_concept_ids]
                    
                    for metric_name in self.summary_metrics.get('binary', {}).keys():
                        key = f"binary_{metric_name}"
                        if key in metric_collection:
                            # Update with all binary concepts at once
                            for c_hat_i, c_true_i in zip(binary_hats, binary_trues):
                                metric_collection[key](c_hat_i, c_true_i)
                
                # Categorical group (average over concepts)
                if self.categorical_concept_ids:
                    for c_id in self.categorical_concept_ids:
                        concept_tensor = concept_tensors[c_id]
                        c_true_i = c_true[:, c_id].long()
                        
                        for metric_name in self.summary_metrics.get('categorical', {}).keys():
                            key = f"categorical_{metric_name}"
                            if key in metric_collection:
                                metric_collection[key](concept_tensor, c_true_i)
                
                # Regression group
                if self.regression_concept_ids:
                    regression_hats = [concept_tensors[i] for i in self.regression_concept_ids]
                    regression_trues = [c_true[:, i].unsqueeze(1) for i in self.regression_concept_ids]
                    
                    for metric_name in self.summary_metrics.get('regression', {}).keys():
                        key = f"regression_{metric_name}"
                        if key in metric_collection:
                            # Update with all regression concepts at once
                            for c_hat_i, c_true_i in zip(regression_hats, regression_trues):
                                metric_collection[key](c_hat_i, c_true_i)
        
        # Update per-concept metrics
        if self.enable_perconcept_metrics:
            if not self.is_nested:
                # Dense format: each concept is a single column
                for c_id, concept_name in enumerate(self.concept_names):
                    c_hat_i = c_hat[:, c_id:c_id+1]
                    c_true_i = c_true[:, c_id:c_id+1]
                    
                    # Update all metrics for this concept
                    for metric_name in self.perconcept_metrics[c_id].keys():
                        key = f"{concept_name}_{metric_name}"
                        if key in metric_collection:
                            task = self.tasks[c_id]
                            
                            if task == 'classification':
                                metric_collection[key](c_hat_i, c_true_i.float())
                            elif task == 'regression':
                                metric_collection[key](c_hat_i, c_true_i)
            else:
                # Nested format: concepts have different sizes
                concept_tensors = torch.split(c_hat, self.cardinalities, dim=1)
                
                for c_id, (concept_name, concept_tensor) in enumerate(zip(self.concept_names, concept_tensors)):
                    c_true_i = c_true[:, c_id]
                    
                    # Update all metrics for this concept
                    for metric_name in self.perconcept_metrics[c_id].keys():
                        key = f"{concept_name}_{metric_name}"
                        if key in metric_collection:
                            task = self.tasks[c_id]
                            card = self.cardinalities[c_id]
                            
                            if task == 'classification' and card == 1:
                                metric_collection[key](concept_tensor, c_true_i.float().unsqueeze(1))
                            elif task == 'classification' and card > 1:
                                metric_collection[key](concept_tensor, c_true_i.long())
                            elif task == 'regression':
                                metric_collection[key](concept_tensor, c_true_i.unsqueeze(1))
          
    def log_metrics(self, metrics, **kwargs):
        self.log_dict(metrics, 
                      on_step=False, 
                      on_epoch=True, 
                      logger=True, 
                      prog_bar=False, 
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        self.log(name + "_loss",
                 loss.detach(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 prog_bar=True,
                 **kwargs)
    
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     # add batch_size to batch
    #     batch['batch_size'] = batch['x'].shape[0]
    #     return batch

    def update_and_log_metrics(self, step, c_hat, c, batch_size):
        """Update and log metrics for the current step."""
        collection = getattr(self, f"{step}_metrics")
        
        if len(collection) == 0:
            return  # No metrics configured
        
        # Update metrics using unified approach
        self._update_metrics(c_hat, c, collection)
        
        # Compute and log results
        results = collection.compute()
        if results:
            formatted_results = {f"{step}/{k}": v for k, v in results.items()}
            self.log_metrics(formatted_results, batch_size=batch_size)

    # def forward(self, *args, **kwargs):
    
    # def predict(self, *args, **kwargs):
    #     h = self.model(*args, **kwargs)
    #     out = self.train_inference.query(h, model=self.model, **kwargs)
    #     return out

    def _unpack_batch(self, batch):
        inputs = batch['inputs']
        concepts = batch['concepts']
        transform = batch.get('transform')
        return inputs, concepts, transform

    def predict_batch(self, 
                      batch, 
                      preprocess: bool = False, 
                      postprocess: bool = True,
                      **forward_kwargs):
        inputs, concepts, transform = self._unpack_batch(batch)

        # apply batch preprocessing
        if preprocess:
            for key, transf in transform.items():
                if key in inputs:
                    inputs[key] = transf.transform(inputs[key])
        if forward_kwargs is None:
            forward_kwargs = dict()
        
        # inference query
        # TODO: train interventions
        if self.train_inference_engine is None:
            # assume the full inference is implemented in the model forward
            out = self.model(**inputs)
        else:
            # model forward (just backbone)
            features = self.model(**inputs)
            # inference
            # TODO: add option to semi-supervise a subset of concepts
            out = self.train_inference_engine.query(self.concept_names, 
                                                    evidence={'emb': features})

        # apply batch postprocess
        # TODO: implement scaling only for continuous / regression concepts 
        # if postprocess:
        #     transf = transform.get('c')
        #     if transf is not None:
        #         out = transf.inverse_transform(out)
        return out


    def shared_step(self, batch, step):
        c = c_loss = batch['concepts']['c']
        out = self.predict_batch(batch, 
                                 preprocess=self.preprocess_inputs, 
                                 postprocess= not self.scale_concepts)
        c_hat_loss = self.model.filter_output_for_loss(out)
        c_hat = self.model.filter_output_for_metric(out)
        if self.scale_concepts:
            raise NotImplementedError("Scaling of concepts is not implemented yet.")
            # c_loss = batch.transform['c'].transform(c)
            # c_hat = batch.transform['c'].inverse_transform(c_hat)

        # Compute loss   
        loss = self._compute_loss(c_hat_loss, c_loss)
    
        # Logging
        batch_size = batch['inputs']['x'].size(0)
        self.update_and_log_metrics(step, c_hat, c, batch_size)
        self.log_loss(step, loss, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, step='train')
        if torch.isnan(loss).any():
            print(f"Loss is 'nan' at epoch: {self.current_epoch}, batch: {batch_idx}")
        return loss

    def validation_step(self, batch):
        loss = self.shared_step(batch, step='val')
        return loss
    
    def test_step(self, batch):
        loss = self.shared_step(batch, step='test')
        
        # TODO: test-time interventions
        # self.test_intervention(batch)
        # if 'Qualified' in self.c_names:
        #     self.test_intervention_fairness(batch)
        return loss



    # def on_train_epoch_end(self):
    #     # Set the current epoch for SCBM and update the list of concept probs for computing the concept percentiles
    #     if type(self.model).__name__ == 'SCBM':
    #         self.model.training_epoch = self.current_epoch
    #         # self.model.concept_pred = torch.cat(self.model.concept_pred_tmp, dim=0) 
    #         # self.model.concept_pred_tmp = []     

    # def on_test_epoch_end(self):
    #     # baseline task accuracy
    #     y_baseline = self.test_y_metrics['y_accuracy'].compute().item()
    #     print(f"Baseline task accuracy: {y_baseline}")
    #     pickle.dump({'_baseline':y_baseline}, open(f'results/y_accuracy.pkl', 'wb'))

    #     # baseline concept accuracy
    #     c_baseline = {}
    #     for k, metric in self.test_c_metrics.items():
    #         k = _remove_prefix(k, self.test_c_metrics.prefix)
    #         c_baseline[k] = metric.compute().item()
    #         print(f"Baseline concept accuracy for {k}: {c_baseline[k]}")
    #     pickle.dump(c_baseline, open(f'results/c_accuracy.pkl', 'wb'))

    #     if self.model.has_concepts:
    #         # task accuracy after invervention on each individual concept
    #         y_int = {}
    #         for k, metric in self.test_intervention_single_y.items():
    #             c_name = _remove_prefix(k, self.test_intervention_single_y.prefix)
    #             y_int[c_name] = metric.compute().item()
    #             print(f"Task accuracy after intervention on {c_name}: {y_int[c_name]}")
    #         pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

    #         # task accuracy after intervention of each policy level
    #         y_int = {}
    #         for k, metric in self.test_intervention_level_y.items():
    #             level = _remove_prefix(k, self.test_intervention_level_y.prefix)
    #             y_int[level] = metric.compute().item()
    #             print(f"Task accuracy after intervention on {level}: {y_int[level]}")
    #         pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

    #         # individual concept accuracy after intervention of each policy level
    #         c_int = {}
    #         for k, metric in self.test_intervention_level_c.items():
    #             level = _remove_prefix(k, self.test_intervention_level_c.prefix)
    #             c_int[level] = metric.compute().item()
    #             print(f"Concept accuracy after intervention on {level}: {c_int[level]}")
    #         pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))

    #         # save graph and concepts
    #         pickle.dump({'concepts':self.c_names,
    #                      'policy':self.test_interv_policy}, open("graph.pkl", 'wb'))
            
    #         pickle.dump({'policy':self.test_interv_policy}, open("policy.pkl", 'wb'))

    def configure_optimizers(self):
        """"""
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
 