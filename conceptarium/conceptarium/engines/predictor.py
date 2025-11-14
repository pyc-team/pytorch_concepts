from typing import Optional, Mapping, Type, Callable, Union
import warnings

import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from torch_concepts import AxisAnnotation

from ..utils import instantiate_from_string


class Predictor(pl.LightningModule):
    def __init__(self,
                model: nn.Module,
                loss: Mapping,
                metrics: Mapping,
                preprocess_inputs: bool = False,
                scale_concepts: bool = False,
                enable_summary_metrics: bool = True,
                enable_perconcept_metrics: Union[bool, list] = False,
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
        self.types = [metadata[name]['type'] for name in self.concept_names]
        self.cardinalities = cardinalities

        self.type_groups = self.concept_annotations.groupby_metadata('type', layout='indices')

        # group concepts by type
        discrete_concept_idx = self.type_groups.get('discrete', [])
        self.binary_concept_idx = [idx for idx in discrete_concept_idx if self.cardinalities[idx] == 1]
        self.categorical_concept_idx = [idx for idx in discrete_concept_idx if self.cardinalities[idx] > 1]
        self.continuous_concept_idx = self.type_groups.get('continuous', [])

        # Pre-compute tensor-slicing indices for each type
        self.cumulative_indices = [0] + list(torch.cumsum(torch.tensor(cardinalities), dim=0).tolist())

        # Binary
        self.binary_idx = []
        for c_id in self.binary_concept_idx:
            self.binary_idx.extend(range(self.cumulative_indices[c_id], self.cumulative_indices[c_id + 1]))
        
        # Categorical
        self.categorical_idx = []
        for c_id in self.categorical_concept_idx:
            self.categorical_idx.extend(range(self.cumulative_indices[c_id], self.cumulative_indices[c_id + 1]))
        
        # Continuous
        self.continuous_idx = []
        for c_id in self.continuous_concept_idx:
            self.continuous_idx.extend(range(self.cumulative_indices[c_id], self.cumulative_indices[c_id + 1]))

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
        types = [c_meta['type'] for _, c_meta in metadata.items()]
        
        # Categorize concepts by type and cardinality
        is_binary = [t == 'discrete' and card == 1 for t, card in zip(types, cardinalities)]
        is_categorical = [t == 'discrete' and card > 1 for t, card in zip(types, cardinalities)]
        is_continuous = [t == 'continuous' for t in types]
        
        has_binary = any(is_binary)
        has_categorical = any(is_categorical)
        has_continuous = any(is_continuous)
        all_same_type = all(t == types[0] for t in types)
        
        # Determine required collection items
        needs_binary = has_binary
        needs_categorical = has_categorical
        needs_continuous = has_continuous
        
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
        binary = get_item(['discrete', 'binary'])
        categorical = get_item(['discrete', 'categorical'])
        continuous = get_item(['continuous'])
        
        # Validation rules
        errors = []
        
        # Check nested/dense compatibility
        if all(is_binary):
            if annotations.is_nested:
                errors.append("Annotations for all-binary concepts should NOT be nested.")
            if not all_same_type:
                errors.append("Annotations for all-binary concepts should share the same type.")
        
        elif all(is_categorical):
            if not annotations.is_nested:
                errors.append("Annotations for all-categorical concepts should be nested.")
            if not all_same_type:
                errors.append("Annotations for all-categorical concepts should share the same type.")
        
        elif all(is_continuous):
            if annotations.is_nested:
                errors.append("Annotations for all-continuous concepts should NOT be nested.")
        
        elif has_binary or has_categorical:
            if not annotations.is_nested:
                errors.append("Annotations for mixed concepts should be nested.")
        
        # Check required items are present
        if needs_binary and binary is None:
            errors.append(f"{collection_name} missing 'discrete.binary' for binary concepts.")
        if needs_categorical and categorical is None:
            errors.append(f"{collection_name} missing 'discrete.categorical' for categorical concepts.")
        if needs_continuous and continuous is None:
            errors.append(f"{collection_name} missing 'continuous' for continuous concepts.")
        
        if errors:
            raise ValueError(f"{collection_name} validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # Warnings for unused items
        if not needs_binary and binary is not None:
            warnings.warn(f"Binary {collection_name} will be ignored (no binary concepts).")
        if not needs_categorical and categorical is not None:
            warnings.warn(f"Categorical {collection_name} will be ignored (no categorical concepts).")
        if not needs_continuous and continuous is not None:
            warnings.warn(f"continuous {collection_name} will be ignored (no continuous concepts).")
        
        # Log configuration
        concept_types = []
        if has_binary and has_categorical:
            concept_types.append("mixed discrete")
        elif has_binary:
            concept_types.append("all binary")
        elif has_categorical:
            concept_types.append("all categorical")
        
        if has_continuous:
            concept_types.append("continuous" if not (has_binary or has_categorical) else "with continuous")
        
        print(f"{collection_name} configuration validated ({', '.join(concept_types)}):")
        print(f"  Binary (card=1): {binary if needs_binary else 'unused'}")
        print(f"  Categorical (card>1): {categorical if needs_categorical else 'unused'}")
        print(f"  continuous: {continuous if needs_continuous else 'unused'}")
        
        # Return only needed items (others set to None)
        return (binary if needs_binary else None,
                categorical if needs_categorical else None,
                continuous if needs_continuous else None)
    
    def _setup_losses(self, loss_config: Mapping):
        """Setup and instantiate loss functions."""
        # Validate and extract needed losses
        binary_cfg, categorical_cfg, continuous_cfg = self._check_collection(
            self.concept_annotations, loss_config, 'loss'
        )
        
        # Instantiate loss functions
        self.binary_loss_fn = instantiate_from_string(binary_cfg['path'], **binary_cfg.get('kwargs', {})) if binary_cfg else None
        self.categorical_loss_fn = instantiate_from_string(categorical_cfg['path'], **categorical_cfg.get('kwargs', {})) if categorical_cfg else None
        self.continuous_loss_fn = instantiate_from_string(continuous_cfg['path'], **continuous_cfg.get('kwargs', {})) if continuous_cfg else None

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
        binary_metrics_cfg, categorical_metrics_cfg, continuous_metrics_cfg = self._check_collection(
            self.concept_annotations, metrics_config, 'metrics'
        )
        
        # Initialize metric storage
        summary_metrics = {}
        perconcept_metrics = []
        
        # Setup summary metrics (one per type group)
        if self.enable_summary_metrics:
            if binary_metrics_cfg:
                summary_metrics['binary'] = self._instantiate_metric_dict(binary_metrics_cfg)
            
            if categorical_metrics_cfg:
                # For categorical, we'll average over individual concept metrics
                self.max_card = max([self.cardinalities[i] for i in self.categorical_concept_idx])
                summary_metrics['categorical'] = self._instantiate_metric_dict(
                    categorical_metrics_cfg, 
                    num_classes=self.max_card
                )
            
            if continuous_metrics_cfg:
                summary_metrics['continuous'] = self._instantiate_metric_dict(continuous_metrics_cfg)
        
        # Setup per-concept metrics (one per concept)
        perconcept_metrics = {}
        if self.enable_perconcept_metrics:
            if isinstance(self.enable_perconcept_metrics, bool):
                self.concepts_to_trace = self.concept_names
            elif isinstance(self.enable_perconcept_metrics, list):
                self.concepts_to_trace = self.enable_perconcept_metrics
            else:
                raise ValueError("enable_perconcept_metrics must be either a bool or a list of concept names.")
            for concept_name in self.concepts_to_trace:
                c_id = self.concept_names.index(concept_name)
                c_type = self.types[c_id]
                card = self.cardinalities[c_id]
                
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

    def _set_metrics(self, summary_metrics: Mapping = None, perconcept_metrics: Mapping = None):
        """Create MetricCollection for train/val/test from summary and per-concept metrics."""
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
                         continuous_fn: Optional[Callable],
                         is_loss: bool) -> Union[torch.Tensor, None]:
        """
        Apply loss or metric functions by looping over concept groups.
        
        Args:
            c_hat: Predicted concepts
            c_true: Ground truth concepts
            binary_fn: Function to apply to binary concepts
            categorical_fn: Function to apply to categorical concepts
            continuous_fn: Function to apply to continuous concepts
            
        Returns:
            For losses: scalar tensor
            For metrics: None (metrics are updated in-place)
        """
        if is_loss:
            loss = 0.0

        if binary_fn:
            c_hat_binary = c_hat[:, self.binary_idx]
            c_true_binary = c_true[:, self.binary_concept_idx].float()
            if is_loss:
                loss += binary_fn(c_hat_binary, c_true_binary)
            else:
                binary_fn.update(c_hat_binary, c_true_binary)

        if categorical_fn:
            # Pad all tensors to max cardinality and stack
            # FIXME: optimize this operation, could this for loop be avoided?
            split_tuple = torch.split(c_hat[:, self.categorical_idx], 
                                      [self.cardinalities[i] for i in self.categorical_concept_idx], dim=1)
            padded_logits = [
                torch.nn.functional.pad(logits, (0, self.max_card - logits.shape[1]), value=float('-inf'))
                for logits in split_tuple
            ]
            c_hat_group = torch.cat(padded_logits, dim=0)
            c_true_group = c_true[:, self.categorical_concept_idx].T.reshape(-1).long()
            
            if is_loss:
                loss += categorical_fn(c_hat_group, c_true_group)
            else:
                categorical_fn.update(c_hat_group, c_true_group)

        if continuous_fn:
            # TODO: verify correctness
            c_hat_continuous = c_hat[:, self.continuous_idx]
            c_true_continuous = c_true[:, self.continuous_concept_idx]
            if is_loss:
                loss += continuous_fn(c_hat_continuous, c_true_continuous)
            else:
                continuous_fn.update(c_hat_continuous, c_true_continuous)

        if is_loss:  
            return loss
        else:
            return None

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
            self.continuous_loss_fn, 
            is_loss=True
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
        for key in metric_collection:

            # Update summary metrics (compute metrics relative to each group)
            if self.enable_summary_metrics:
                if 'SUMMARY-binary_' in key and self.binary_concept_idx:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=metric_collection[key],
                        categorical_fn=None,
                        continuous_fn=None,
                        is_loss=False
                    )
                    continue
                
                elif 'SUMMARY-categorical_' in key and self.categorical_concept_idx:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=None,
                        categorical_fn=metric_collection[key],
                        continuous_fn=None,
                        is_loss=False
                    )
                    continue
                
                elif 'SUMMARY-continuous_' in key and self.continuous_concept_idx:
                    self._apply_fn_by_type(
                        c_hat, c_true,
                        binary_fn=None,
                        categorical_fn=None,
                        continuous_fn=metric_collection[key],
                        is_loss=False
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
                card = self.cardinalities[c_id]

                start_idx = self.cumulative_indices[c_id]
                end_idx = self.cumulative_indices[c_id + 1]

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

    def update_and_log_metrics(self, step, c_hat, c, batch_size):
        """Update and log metrics for the current step."""
        collection = getattr(self, f"{step}_metrics")
        
        if len(collection) == 0:
            return  # No metrics configured
        
        # Update metrics by groups and per-concept
        self._update_metrics(c_hat.detach(), c, collection)
        # log metrics
        self.log_metrics(collection, batch_size=batch_size)

    def _unpack_batch(self, batch):
        inputs = batch['inputs']
        concepts = batch['concepts']
        transform = batch.get('transform')
        inputs, concepts = self.model.preprocess_batch(inputs, concepts)
        return inputs, concepts, transform

    def predict_batch(self, 
                      batch, 
                      preprocess: bool = False, 
                      postprocess: bool = True,
                      **forward_kwargs):
        inputs, _, transform = self._unpack_batch(batch)

        # apply batch preprocessing
        if preprocess:
            for key, transf in transform.items():
                if key in inputs:
                    inputs[key] = transf.transform(inputs[key])
        if forward_kwargs is None:
            forward_kwargs = dict()
        
        # model forward (containing inference query)
        # TODO: implement train interventions using the context manager 'with ...'
        # TODO: add option to semi-supervise a subset of concepts
        # TODO: handle backbone kwargs when present
        out = self.model.forward(x=inputs['x'],
                                 query=self.concept_names, 
                                 **forward_kwargs)
            
        # # TODO: implement scaling only for continuous concepts 
        # # apply batch postprocess
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
            # # TODO: implement scaling only for continuous concepts 
            # c_loss = batch.transform['c'].transform(c)
            # c_hat = batch.transform['c'].inverse_transform(c_hat)

        # Compute loss   
        loss = self._compute_loss(c_hat_loss, c_loss)

        # Logging
        batch_size = batch['inputs']['x'].size(0)
        self.log_loss(step, loss, batch_size=batch_size)
        self.update_and_log_metrics(step, c_hat, c, batch_size)

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
 