"""Loss functions for concept-based models."""
from typing import List, Mapping
import torch
from torch import nn

from ...nn.modules.utils import GroupConfig
from ...annotations import Annotations, AxisAnnotation
from ...utils import instantiate_from_string
from ...nn.modules.utils import check_collection


def get_concept_task_idx(annotations: AxisAnnotation, concepts: List[str], tasks: List[str]):
    """Get concept and task indices at both concept-level and logit-level."""
    # Concept-level indices
    concepts_idxs = [annotations.get_index(name) for name in concepts]
    tasks_idxs = [annotations.get_index(name) for name in tasks]
    
    # Logit-level indices using cached get_slice
    concepts_logits = annotations.get_slice(concepts)
    tasks_logits = annotations.get_slice(tasks)
    
    return concepts_idxs, tasks_idxs, concepts_logits, tasks_logits

class ConceptLoss(nn.Module):
    """
    Concept loss for concept-based models.

    Automatically routes to appropriate loss functions based on concept types
    (binary, categorical, continuous) using annotation metadata.

    Args:
        annotations (Annotations): Concept annotations with metadata including
            type information for each concept.
        fn_collection (GroupConfig): Loss function configuration per concept type.
            Keys should be 'binary', 'categorical', and/or 'continuous'.

    Example:
        >>> from torch_concepts.nn import ConceptLoss
        >>> from torch_concepts import GroupConfig, Annotations, AxisAnnotation
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> from torch.distributions import Bernoulli, Categorical
        >>> 
        >>> # Define annotations
        >>> ann = Annotations({1: AxisAnnotation(
        ...     labels=['is_round', 'color'],
        ...     cardinalities=[1, 3],
        ...     metadata={
        ...         'is_round': {'type': 'discrete', 'distribution': Bernoulli},
        ...         'color': {'type': 'discrete', 'distribution': Categorical}
        ...     }
        ... )})
        >>> 
        >>> # Configure loss functions
        >>> loss_config = GroupConfig(
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss()
        ... )
        >>> loss_fn = ConceptLoss(ann[1], loss_config)
        >>> 
        >>> # Compute loss
        >>> predictions = torch.randn(2, 4)  # 1 binary + 3 categorical logits
        >>> targets = torch.cat([
        ...     torch.randint(0, 2, (2, 1)),  # binary target
        ...     torch.randint(0, 3, (2, 1))   # categorical target
        ... ], dim=1)
        >>> loss = loss_fn(predictions, targets)
    """
    def __init__(self, annotations: Annotations, fn_collection: GroupConfig):
        super().__init__()
        annotations = annotations.get_axis_annotation(axis=1)
        self.fn_collection = check_collection(annotations, fn_collection, 'loss')
        
        # Use cached type_groups from AxisAnnotation
        self.groups = annotations.type_groups
        self.cardinalities = annotations.cardinalities

        # For categorical loss, precompute max cardinality for padding
        if self.fn_collection.get('categorical'):
            cat_idx = self.groups['categorical']['concept_idx']
            self.max_card = max([self.cardinalities[i] for i in cat_idx])

        if self.fn_collection.get('continuous'):
            cont_idx = self.groups['continuous']['concept_idx']
            self.max_dim = max([self.cardinalities[i] for i in cont_idx])

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
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels/values.
            
        Returns:
            torch.Tensor: Total computed loss (scalar).
        """
        total_loss = 0.0
        
        # Binary concepts
        if self.fn_collection.get('binary'): 
            binary_logits = input[:, self.groups['binary']['logits_idx']]
            binary_targets = target[:, self.groups['binary']['concept_idx']].float()
            total_loss += self.fn_collection['binary'](binary_logits, binary_targets)
        
        # Categorical concepts
        if self.fn_collection.get('categorical'):
            split_tuple = torch.split(
                input[:, self.groups['categorical']['logits_idx']], 
                [self.cardinalities[i] for i in self.groups['categorical']['concept_idx']], 
                dim=1
            )
            padded_logits = [
                nn.functional.pad(
                    logits, 
                    (0, self.max_card - logits.shape[1]), 
                    value=float('-inf')
                ) for logits in split_tuple
            ]
            cat_logits = torch.cat(padded_logits, dim=0)
            cat_targets = target[:, self.groups['categorical']['concept_idx']].T.reshape(-1).long()
            
            total_loss += self.fn_collection['categorical'](cat_logits, cat_targets)
        
        # Continuous concepts
        if self.fn_collection.get('continuous'):
            raise NotImplementedError("Continuous concepts not yet implemented.")
        
        return total_loss


class WeightedConceptLoss(nn.Module):
    """
    Weighted concept loss for concept-based models.

    Computes a weighted combination of concept and task losses.

    Args:
        annotations (Annotations): Annotations object with concept metadata.
        fn_collection (GroupConfig): Loss function configuration.
        concept_weight (float): Weight for concept loss
        task_weight (float): Weight for task loss
        task_names (List[str]): List of task concept names.

    Example:
        >>> from torch_concepts.nn.modules.loss import WeightedConceptLoss
        >>> from torch_concepts.nn.modules.utils import GroupConfig
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> from torch_concepts.annotations import AxisAnnotation, Annotations
        >>> ann = Annotations({1: AxisAnnotation(labels=['c1', 'c2', 'task'], cardinalities=[1, 3, 1])})
        >>> fn = GroupConfig(binary=BCEWithLogitsLoss(), categorical=CrossEntropyLoss())
        >>> loss_fn = WeightedConceptLoss(ann, fn, weight=0.7, task_names=['task'])
        >>> input = torch.randn(2, 5)
        >>> target = torch.randint(0, 2, (2, 3))
        >>> loss = loss_fn(input, target)
    """
    def __init__(
        self, 
        annotations: Annotations, 
        fn_collection: GroupConfig,
        concept_weight: float,
        task_weight: float,
        task_names: List[str]
    ):
        super().__init__()
        self.concept_weight = concept_weight
        self.task_weight = task_weight
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
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels/values.
        
        Returns:
            torch.Tensor: Weighted combination of concept and task losses (scalar).
        """
        concept_input = input[:, self.input_c_idx]
        concept_target = target[:, self.target_c_idx]
        task_input = input[:, self.input_t_idx]
        task_target = target[:, self.target_t_idx]
        
        c_loss = self.concept_loss(concept_input, concept_target)
        t_loss = self.task_loss(task_input, task_target)
        
        return c_loss * self.concept_weight + t_loss * self.task_weight


class CMRLoss(nn.Module):
    """
    Loss for Concept-based Memory Reasoner (CMR).

    Implements the objective used in CMR examples:
    - concept loss on concept logits
    - task loss without reconstruction term
    - task loss with reconstruction term
    - blended task objective that applies reconstruction-aware loss on
      positive targets and standard loss on negative targets

    Args:
        concept_weight: Weight applied to concept loss.
        task_weight: Weight applied to blended task loss.
    """
    def __init__(
        self,
        concept_weight: float = 1.0,
        task_weight: float = 1.0,
    ):
        super().__init__()
        self.concept_loss_fn = nn.BCEWithLogitsLoss()
        self.task_loss_fn = nn.BCELoss(reduction='none')
        self.concept_weight = concept_weight
        self.task_weight = task_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"concept_loss_fn={self.concept_loss_fn.__class__.__name__}, "
            f"task_loss_fn={self.task_loss_fn.__class__.__name__}, "
            f"concept_weight={self.concept_weight}, "
            f"task_weight={self.task_weight})"
        )

    def _compute_explicit(
        self,
        concept_input: torch.Tensor,
        concept_target: torch.Tensor,
        task_input: torch.Tensor,
        task_input_with_rec: torch.Tensor,
        task_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CMR objective.

        Args:
            concept_input: Concept logits.
            concept_target: Concept targets.
            task_input: Task probabilities without reconstruction term.
            task_input_with_rec: Task probabilities with reconstruction term.
            task_target: Task targets.

        Returns:
            Scalar CMR loss.
        """
        concept_target = concept_target.float()
        task_target = task_target.float()

        concept_loss = self.concept_loss_fn(concept_input, concept_target)

        task_loss_no_rec = self.task_loss_fn(task_input, task_target)
        task_loss_rec = self.task_loss_fn(task_input_with_rec, task_target)

        if task_loss_no_rec.shape != task_target.shape or task_loss_rec.shape != task_target.shape:
            raise ValueError(
                "task_loss_fn must return elementwise losses with the same "
                "shape as task_target (use reduction='none')."
            )

        blended_task_loss = (task_target * task_loss_rec + (1 - task_target) * task_loss_no_rec).mean()

        return self.concept_weight * concept_loss + self.task_weight * blended_task_loss

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute CMR loss from explicit CMR tensors only."""
        explicit_keys = {
            'concept_input',
            'concept_target',
            'task_input',
            'task_input_with_rec',
            'task_target',
        }

        if explicit_keys.issubset(kwargs.keys()):
            return self._compute_explicit(
                concept_input=kwargs['concept_input'],
                concept_target=kwargs['concept_target'],
                task_input=kwargs['task_input'],
                task_input_with_rec=kwargs['task_input_with_rec'],
                task_target=kwargs['task_target'],
            )

        raise ValueError(
            "CMRLoss.forward requires explicit CMR tensors: "
            "concept_input, concept_target, task_input, task_input_with_rec, task_target."
        )