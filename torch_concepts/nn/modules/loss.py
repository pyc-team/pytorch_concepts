"""Loss functions for concept-based models."""
from typing import List, Mapping
import torch
from torch import nn

from ...nn.modules.utils import GroupConfig
from ...annotations import Annotations, AxisAnnotation
from ...utils import instantiate_from_string
from ...nn.modules.utils import check_collection, get_concept_groups


def get_concept_task_idx(annotations: AxisAnnotation, concepts: List[str], tasks: List[str]):
    # Concept-level indices: position in concept list
    concepts_idxs = [annotations.get_index(name) for name in concepts]
    tasks_idxs = [annotations.get_index(name) for name in tasks]
    cumulative_indices = [0] + list(torch.cumsum(torch.tensor(annotations.cardinalities), dim=0).tolist())

    # Logit-level indices: position in flattened tensor (accounting for cardinality)
    concepts_endogenous = []
    for idx in concepts_idxs:
        concepts_endogenous.extend(range(cumulative_indices[idx], cumulative_indices[idx + 1]))

    tasks_endogenous = []
    for idx in tasks_idxs:
        tasks_endogenous.extend(range(cumulative_indices[idx], cumulative_indices[idx + 1]))
    
    return concepts_idxs, tasks_idxs, concepts_endogenous, tasks_endogenous

class ConceptLoss(nn.Module):
    def __init__(self, annotations: Annotations, fn_collection: GroupConfig):
        super().__init__()
        annotations = annotations.get_axis_annotation(axis=1)
        self.fn_collection = check_collection(annotations, fn_collection, 'loss')
        self.groups = get_concept_groups(annotations)
        self.cardinalities = annotations.cardinalities

        # For categorical loss, precompute max cardinality for padding
        if self.fn_collection.get('categorical'):
            self.max_card = max([self.cardinalities[i] for i in self.groups['categorical_idx']])

        if self.fn_collection.get('continuous'):
            self.max_dim = max([self.cardinalities[i] for i in self.groups['continuous_idx']])

    def __repr__(self) -> str:
        types = ['binary', 'categorical', 'continuous']
        parts = []
        for t in types:
            loss = self.fn_collection.get(t)
            if loss:
                if isinstance(loss, nn.Module):
                    name = loss.__class__.__name__
                elif isinstance(loss, (tuple, list)):
                    name = loss[0].__name__
                else:
                    name = loss.__name__
                parts.append(f"{t}={name}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute total loss across all concept types.
        
        Splits inputs and targets by concept type, computes individual losses,
        and sums them to get the total loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (endogenous or values).
            targets (torch.Tensor): Ground truth labels/values.
            
        Returns:
            Tenso: Total computed loss.
        """
        total_loss = 0.0
        
        # Binary concepts
        if self.fn_collection.get('binary'):
            binary_endogenous = input[:, self.groups['binary_endogenous_idx']]
            binary_targets = target[:, self.groups['binary_idx']].float()
            total_loss += self.fn_collection['binary'](binary_endogenous, binary_targets)
        
        # Categorical concepts
        if self.fn_collection.get('categorical'):
            split_tuple = torch.split(
                input[:, self.groups['categorical_endogenous_idx']], 
                [self.cardinalities[i] for i in self.groups['categorical_idx']], 
                dim=1
            )
            padded_endogenous = [
                nn.functional.pad(
                    endogenous, 
                    (0, self.max_card - endogenous.shape[1]), 
                    value=float('-inf')
                ) for endogenous in split_tuple
            ]
            cat_endogenous = torch.cat(padded_endogenous, dim=0)
            cat_targets = target[:, self.groups['categorical_idx']].T.reshape(-1).long()
            
            total_loss += self.fn_collection['categorical'](cat_endogenous, cat_targets)
        
        # Continuous concepts
        if self.fn_collection.get('continuous'):
            raise NotImplementedError("Continuous concepts not yet implemented.")
        
        return total_loss


class WeightedConceptLoss(nn.Module):
    """Concept loss with separate weighting for each concept type.
    
    Args:
        annotations (Annotations): Annotations object with concept metadata.
        fn_collection (Mapping): Loss function configuration.
        weight (float): Weight for concept loss; (1 - weight) is for task loss.
        task_names (List[str]): List of task concept names.
    """
    def __init__(
        self, 
        annotations: Annotations, 
        fn_collection: GroupConfig,
        weight: float,
        task_names: List[str]
    ):
        super().__init__()
        self.weight = weight
        self.fn_collection = fn_collection
        annotations = annotations.get_axis_annotation(axis=1)
        concept_names = [name for name in annotations.labels if name not in task_names]
        task_annotations = Annotations({1:annotations.subset(task_names)})
        concept_annotations = Annotations({1:annotations.subset(concept_names)})

        self.concept_loss = ConceptLoss(concept_annotations, fn_collection)
        self.task_loss = ConceptLoss(task_annotations, fn_collection)
        self.target_c_idx, self.target_t_idx, self.input_c_idx, self.input_t_idx = get_concept_task_idx(
            annotations, concept_names, task_names
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fn_collection={self.fn_collection})"
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss for concepts and tasks.
        
        Args:
            inputs (torch.Tensor): Model predictions (endogenous or values).
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
