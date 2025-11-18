"""Loss functions for concept-based models."""

import torch

class WeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
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


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
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
        

class WeightedMSELoss(torch.nn.MSELoss):
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