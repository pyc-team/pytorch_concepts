from abc import abstractmethod
from ..base.learner import BaseLearner 


class JointLearner(BaseLearner):
    """
    Joint training engine for concept-based models.

    Extends BaseLearner to support joint training of all concepts and tasks.

    Example:
        >>> from torch_concepts.nn.modules.high.learners.joint import JointLearner
        >>> learner = JointLearner(loss=None, metrics=None)
    """
    def __init__(self,**kwargs):
        super(JointLearner, self).__init__(**kwargs)

    @abstractmethod
    def forward(self, x, query, *args, **kwargs):
        """Model forward method to be implemented by subclasses.
        """
        pass
    
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
        if self.loss is not None:
            loss_args = self.filter_output_for_loss(out, c_loss)
            loss = self.loss(**loss_args)
            self.log_loss(step, loss, batch_size=batch_size)

        # --- Update and log metrics ---
        metrics_args = self.filter_output_for_metrics(out, c)
        self.update_and_log_metrics(metrics_args, step, batch_size)
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
 