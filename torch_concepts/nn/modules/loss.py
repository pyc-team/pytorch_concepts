"""Loss functions for concept-based models."""
from typing import List, Mapping
import torch
from torch import nn

from torch_concepts import AxisAnnotation
from torch_concepts.utils import instantiate_from_string
from torch_concepts.nn.modules.utils import check_collection, get_concept_groups

def setup_losses(annotations: AxisAnnotation, loss_config: Mapping):
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
    binary_fn = instantiate_from_string(binary_cfg['path'], **binary_cfg.get('kwargs', {})) if binary_cfg else None
    categorical_fn = instantiate_from_string(categorical_cfg['path'], **categorical_cfg.get('kwargs', {})) if categorical_cfg else None
    continuous_fn = instantiate_from_string(continuous_cfg['path'], **continuous_cfg.get('kwargs', {})) if continuous_cfg else None
    
    return binary_fn, categorical_fn, continuous_fn


def get_concept_task_idx(annotations: AxisAnnotation, concepts: List[str], tasks: List[str]):
    # Concept-level indices: position in concept list
    concepts_idxs = [annotations.get_index(name) for name in concepts]
    tasks_idxs = [annotations.get_index(name) for name in tasks]
    cumulative_indices = [0] + list(torch.cumsum(torch.tensor(annotations.cardinalities), dim=0).tolist())

    # Logit-level indices: position in flattened tensor (accounting for cardinality)
    concepts_logits = []
    for idx in concepts_idxs:
        concepts_logits.extend(range(cumulative_indices[idx], cumulative_indices[idx + 1]))

    tasks_logits = []
    for idx in tasks_idxs:
        tasks_logits.extend(range(cumulative_indices[idx], cumulative_indices[idx + 1]))
    
    return concepts_idxs, tasks_idxs, concepts_logits, tasks_logits

class ConceptLoss(nn.Module):
    def __init__(self, 
                 annotations: AxisAnnotation, 
                 fn_collection: Mapping
    ):
        super().__init__()
        self.binary_fn, self.categorical_fn, self.continuous_fn = setup_losses(annotations, fn_collection)
        self.groups = get_concept_groups(annotations)

        # For categorical loss, precompute max cardinality for padding
        if self.categorical_fn is not None:
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
            cont_preds = input[:, self.groups['continuous_logits']]
            cont_targets = target[:, self.groups['continuous_concepts']]
            total_loss += self.continuous_fn(cont_preds, cont_targets)
        
        return total_loss


class WeightedConceptLoss(nn.Module):
    """Concept loss with separate weighting for each concept type.
    
    Args:
        annotations (Annotations): Annotations object with concept metadata.
        fn_collection (Mapping): Loss function configuration.
        weights (Mapping): Weights for each concept type, e.g.:
            {'binary': 0.5, 'categorical': 0.3, 'continuous': 0.2}
    """
    def __init__(self, 
                 annotations: AxisAnnotation, 
                 fn_collection: Mapping,
                 weight: Mapping,
                 task_names: List[str]
    ):
        super().__init__()
        self.weight = weight
        concept_names = [name for name in annotations.labels if name not in task_names]
        task_annotations = annotations.subset(task_names)
        concept_annotations = annotations.subset(concept_names)
        self.concept_loss = ConceptLoss(concept_annotations, fn_collection)
        self.task_loss = ConceptLoss(task_annotations, fn_collection)
        self.target_c_idx, self.target_t_idx, self.input_c_idx, self.input_t_idx = get_concept_task_idx(
            annotations, concept_names, task_names
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss for concepts and tasks.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits or values).
            targets (torch.Tensor): Ground truth labels/values.
        
        Returns:
            Tensor: Weighted combination of concept and task losses.
        """
        concept_input = input[:, self.input_c_idx]
        concept_target = target[:, self.target_c_idx]
        task_input = input[:, self.input_t_idx]
        task_target = target[:, self.target_t_idx]
        
        c_loss = self.concept_loss(concept_input, concept_target)
        t_loss = self.task_loss(task_input, task_target)
        
        return c_loss * self.weight + t_loss * (1 - self.weight)
