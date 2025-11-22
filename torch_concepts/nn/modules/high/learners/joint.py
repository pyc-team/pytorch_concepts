from abc import abstractmethod
from typing import Mapping, Type, Union, Optional
import torch
from torch import nn

from torch_concepts.annotations import Annotations

from ..base.learner import BaseLearner 


class JointLearner(BaseLearner):
    def __init__(self,
                loss: nn.Module,
                metrics: Mapping,
                annotations: Annotations,
                variable_distributions: Mapping,
                optim_class: Type,
                optim_kwargs: Mapping,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,  
                enable_summary_metrics: Optional[bool] = True,
                enable_perconcept_metrics: Optional[Union[bool, list]] = False,
                **kwargs
    ):
        super(JointLearner, self).__init__(
            loss=loss,
            metrics=metrics,
            annotations=annotations,
            variable_distributions=variable_distributions,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            enable_summary_metrics=enable_summary_metrics,
            enable_perconcept_metrics=enable_perconcept_metrics,
            **kwargs
        )

    def shared_step(self, batch, step):
        """Shared logic for train/val/test steps.
        
        Performs forward pass, loss computation, and metric logging.
        
        Args:
            batch (dict): Batch dictionary from dataloader.
            step (str): One of 'train', 'val', or 'test'.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        inputs, concepts, transforms = self.unpack_batch(batch)
        batch_size = batch['inputs']['x'].size(0)
        c = c_loss = concepts['c']

        # TODO: implement scaling only for continuous concepts 
        # inputs = self.maybe_apply_preprocessing(preprocess_inputs_flag, 
        #                                         inputs, 
        #                                         transforms)

        # --- Model forward ---
        # joint training -> inference on all concepts
        # TODO: implement train interventions using the context manager 'with ...'
        # TODO: add option to semi-supervise a subset of concepts
        # TODO: handle backbone kwargs when present
        out = self.forward(x=inputs['x'], query=self.concept_names)

        # TODO: implement scaling only for continuous concepts 
        # out = self.maybe_apply_postprocessing(not scale_concepts_flag, 
        #                                       out, 
        #                                       transforms)
        # if scale_concepts_flag:
        #     c_loss = batch.transform['c'].transform(c)
        #     c_hat = batch.transform['c'].inverse_transform(c_hat)

        # --- Compute loss ---
        # keys in in_loss_dict must match those expected by loss functions
        in_loss_dict = self.filter_output_for_loss(out, c_loss)
        loss = self.loss_fn(**in_loss_dict)
        self.log_loss(step, loss, batch_size=batch_size)

        # --- Update and log metrics ---
        collection = getattr(self, f"{step}_metrics")
        in_metric_dict = self.filter_output_for_metric(out, c)
        self.update_metrics(in_metric_dict, collection)
        self.log_metrics(collection, batch_size=batch_size)

        return loss

    def training_step(self, batch):
        """Training step called by PyTorch Lightning.
        
        Args:
            batch (dict): Training batch.
            
        Returns:
            torch.Tensor: Training loss.
        """
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
 