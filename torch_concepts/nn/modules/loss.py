"""Loss functions for concept-based models."""
from typing import Mapping
import torch
from torch import nn

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.utils import instantiate_from_string
from torch_concepts.nn.modules.utils import check_collection, get_concept_groups


class ConceptLoss(nn.Module):
    def __init__(self, 
                 annotations: Annotations, 
                 fn_collection: Mapping):
        super().__init__()
        annotations = annotations.get_axis_annotation(1)
        self._setup_losses(annotations, fn_collection)
        self.groups = get_concept_groups(annotations)

    def _setup_losses(self, annotations: AxisAnnotation, loss_config: Mapping):
        """Setup and instantiate loss functions from configuration.
        
        Validates the loss config and creates loss function instances for each
        concept type (binary, categorical, continuous) based on what's needed.
        
        Args:
            loss_config (Mapping): Nested dict with structure:
                {'discrete': {'binary': {...}, 'categorical': {...}}, 
                 'continuous': {...}}
        """
        # Validate and extract needed losses
        binary_cfg, categorical_cfg, continuous_cfg = check_collection(
            annotations, loss_config, 'loss'
        )
        
        # Instantiate loss functions
        self.binary_fn = instantiate_from_string(binary_cfg['path'], **binary_cfg.get('kwargs', {})) if binary_cfg else None
        self.categorical_fn = instantiate_from_string(categorical_cfg['path'], **categorical_cfg.get('kwargs', {})) if categorical_cfg else None
        self.continuous_fn = instantiate_from_string(continuous_cfg['path'], **continuous_cfg.get('kwargs', {})) if continuous_cfg else None
        # For categorical loss, precompute max cardinality for padding
        if categorical_cfg:
            self.cardinalities = annotations.cardinalities
            self.max_card = max([self.cardinalities[i] for i in self.cardinalities])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute total loss across all concept types.
        
        Splits inputs and targets by concept type, computes individual losses,
        and sums them to get the total loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits or values).
            targets (torch.Tensor): Ground truth labels/values.
            
        Returns:
            Tenso: Total computed loss.
        """
        total_loss = 0.0
        
        # Binary concepts
        if self.binary_fn is not None:
            binary_logits = input[:, self.groups['binary_logits']]
            binary_targets = target[:, self.groups['binary_concepts']].float()
            total_loss += self.binary_fn(binary_logits, binary_targets)
        
        # Categorical concepts
        if self.categorical_fn is not None:
            split_tuple = torch.split(input[:, self.groups['categorical_logits']], 
                                      [self.cardinalities[i] for i in self.groups['categorical_concepts']], dim=1)
            padded_logits = [nn.functional.pad(logits, (0, self.max_card - logits.shape[1]), value=float('-inf'))
                             for logits in split_tuple]
            cat_logits = torch.cat(padded_logits, dim=0)
            cat_targets = target[:, self.groups['categorical_concepts']].T.reshape(-1).long()
            
            total_loss += self.categorical_fn(cat_logits, cat_targets)
        
        # Continuous concepts
        if self.continuous_fn is not None:
            cont_preds = input[:, self.groups['continuous_concepts']]
            cont_targets = target[:, self.groups['continuous_concepts']]
            total_loss += self.continuous_fn(cont_preds, cont_targets)
        
        return total_loss







class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Binary Cross-Entropy loss with separate weighting for concepts and tasks.
    
    Computes BCE loss separately for concept predictions and task predictions,
    then combines them with optional weighting. If concept_loss_weight is None,
    returns unweighted sum.
    
    Args:
        concept_loss_weight (float, optional): Weight for concept loss in [0, 1].
            Task loss weight is automatically (1 - concept_loss_weight).
            If None, returns unweighted sum. Defaults to None.
        **kwargs: Additional arguments passed to torch.nn.BCEWithLogitsLoss.
            
    Example:
        >>> loss_fn = WeightedBCEWithLogitsLoss(concept_loss_weight=0.8)
        >>> concept_logits = torch.randn(32, 10)  # 32 samples, 10 concepts
        >>> task_logits = torch.randn(32, 5)      # 32 samples, 5 tasks
        >>> concept_targets = torch.randint(0, 2, (32, 10)).float()
        >>> task_targets = torch.randint(0, 2, (32, 5)).float()
        >>> loss = loss_fn(concept_logits, task_logits, concept_targets, task_targets)
        >>> # loss = 0.8 * BCE(concept) + 0.2 * BCE(task)
    """
    def __init__(self, concept_loss_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.concept_loss_weight = concept_loss_weight
    
    def forward(self, 
                concept_input: torch.Tensor, task_input: torch.Tensor, 
                concept_target: torch.Tensor, task_target: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss for concepts and tasks.
        
        Args:
            concept_input (torch.Tensor): Concept logits (pre-sigmoid).
            task_input (torch.Tensor): Task logits (pre-sigmoid).
            concept_target (torch.Tensor): Concept binary targets.
            task_target (torch.Tensor): Task binary targets.
            
        Returns:
            torch.Tensor: Weighted combination of concept and task losses.
        """
        if self.concept_loss_weight is not None:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return (c_loss * self.concept_loss_weight) + (t_loss * (1 - self.concept_loss_weight))
        else:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return c_loss + t_loss


class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-Entropy loss with separate weighting for concepts and tasks.
    
    Computes CE loss separately for concept predictions and task predictions,
    then combines them with optional weighting. Suitable for multi-class
    classification tasks.
    
    Args:
        concept_loss_weight (float, optional): Weight for concept loss in [0, 1].
            Task loss weight is automatically (1 - concept_loss_weight).
            If None, returns unweighted sum. Defaults to None.
        **kwargs: Additional arguments passed to torch.nn.CrossEntropyLoss.
            
    Example:
        >>> loss_fn = WeightedCrossEntropyLoss(concept_loss_weight=0.6)
        >>> concept_logits = torch.randn(32, 10, 5)  # 32 samples, 10 concepts, 5 classes
        >>> task_logits = torch.randn(32, 3, 8)      # 32 samples, 3 tasks, 8 classes
        >>> concept_targets = torch.randint(0, 5, (32, 10))
        >>> task_targets = torch.randint(0, 8, (32, 3))
        >>> loss = loss_fn(concept_logits, concept_targets, task_logits, task_targets)
    """
    def __init__(self, concept_loss_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.concept_loss_weight = concept_loss_weight
    
    def forward(self, 
                concept_input: torch.Tensor, 
                concept_target: torch.Tensor, 
                task_input: torch.Tensor, 
                task_target: torch.Tensor) -> torch.Tensor:
        """Compute weighted CE loss for concepts and tasks.
        
        Args:
            concept_input (torch.Tensor): Concept logits.
            concept_target (torch.Tensor): Concept class indices.
            task_input (torch.Tensor): Task logits.
            task_target (torch.Tensor): Task class indices.
            
        Returns:
            torch.Tensor: Weighted combination of concept and task losses.
        """
        if self.concept_loss_weight is not None:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return (c_loss * self.concept_loss_weight) + (t_loss * (1 - self.concept_loss_weight))
        else:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return c_loss + t_loss
        

class WeightedMSELoss(nn.MSELoss):
    """Mean Squared Error loss with separate weighting for concepts and tasks.
    
    Computes MSE loss separately for concept predictions and task predictions,
    then combines them with optional weighting. Suitable for regression tasks.
    
    Args:
        concept_loss_weight (float, optional): Weight for concept loss in [0, 1].
            Task loss weight is automatically (1 - concept_loss_weight).
            If None, returns unweighted sum. Defaults to None.
        **kwargs: Additional arguments passed to torch.nn.MSELoss.
            
    Example:
        >>> loss_fn = WeightedMSELoss(concept_loss_weight=0.75)
        >>> concept_preds = torch.randn(32, 10)  # 32 samples, 10 continuous concepts
        >>> task_preds = torch.randn(32, 3)      # 32 samples, 3 continuous tasks
        >>> concept_targets = torch.randn(32, 10)
        >>> task_targets = torch.randn(32, 3)
        >>> loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)
    """
    def __init__(self, concept_loss_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.concept_loss_weight = concept_loss_weight
    
    def forward(self, 
                concept_input: torch.Tensor, 
                concept_target: torch.Tensor, 
                task_input: torch.Tensor, 
                task_target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss for concepts and tasks.
        
        Args:
            concept_input (torch.Tensor): Concept predictions.
            concept_target (torch.Tensor): Concept ground truth values.
            task_input (torch.Tensor): Task predictions.
            task_target (torch.Tensor): Task ground truth values.
            
        Returns:
            torch.Tensor: Weighted combination of concept and task losses.
        """
        if self.concept_loss_weight is not None:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return (c_loss * self.concept_loss_weight) + (t_loss * (1 - self.concept_loss_weight))
        else:
            c_loss = super().forward(concept_input, concept_target)
            t_loss = super().forward(task_input, task_target)
            return c_loss + t_loss