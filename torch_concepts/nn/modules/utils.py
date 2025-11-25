from typing import Mapping, Optional, Tuple, Dict, Union, List
import warnings
import logging
import torch

from ...annotations import AxisAnnotation

logger = logging.getLogger(__name__)

def check_collection(annotations: AxisAnnotation, 
                     collection: Mapping,
                     collection_name: str):
    """Validate loss/metric configurations against concept annotations.
    
    Ensures that:
    1. Required losses/metrics are present for each concept type
    2. Annotation structure (nested vs dense) matches concept types
    3. Unused configurations are warned about
    
    Args:
        annotations (AxisAnnotation): Concept annotations with metadata.
        collection (Mapping): Nested dict of losses or metrics.
        collection_name (str): Either 'loss' or 'metrics' for error messages.
        
    Returns:
        Tuple[Optional[dict], Optional[dict], Optional[dict]]: 
            (binary_config, categorical_config, continuous_config) 
            Only returns configs needed for the actual concept types present.
            
    Raises:
        ValueError: If validation fails (missing required configs, 
            incompatible annotation structure).
            
    Example:
        >>> binary_loss, cat_loss, cont_loss = check_collection(
        ...     self.concept_annotations, 
        ...     loss_config, 
        ...     'loss'
        ... )
    """
    assert collection_name in ['loss', 'metrics'], f"collection_name must be \
        either 'loss' or 'metrics', got '{collection_name}'"

    # Extract annotation properties
    metadata = annotations.metadata
    cardinalities = annotations.cardinalities
    types = [c_meta['type'] for _, c_meta in metadata.items()]
    
    # Categorize concepts by type and cardinality
    is_binary = [x == ('discrete', 1) for x in zip(types, cardinalities)]
    is_categorical = [t == 'discrete' and card > 1 for t, card in zip(types, cardinalities)]
    is_continuous = [t == 'continuous' for t in types]

    # raise error if continuous concepts are present
    if any(is_continuous):
        raise NotImplementedError("Continuous concepts not yet implemented.")
    
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
    
    # TODO: discuss whether to keep these debuggin loggin lines
    # logger.info(f"{collection_name} configuration validated ({', '.join(concept_types)}):")
    # logger.info(f"  Binary (card=1): {binary if needs_binary else 'unused'}")
    # logger.info(f"  Categorical (card>1): {categorical if needs_categorical else 'unused'}")
    # logger.info(f"  continuous: {continuous if needs_continuous else 'unused'}")
    
    # Return only needed items (others set to None)
    return (binary if needs_binary else None,
            categorical if needs_categorical else None,
            continuous if needs_continuous else None)


def get_concept_groups(annotations: AxisAnnotation) -> Dict[str, list]:
    """Compute concept grouping by type for efficient loss/metric computation.
    
    Creates index mappings to slice tensors by concept type. Returns indices at two levels:
    1. Concept-level indices: Position in concept list (e.g., concept 0, 1, 2...)
    2. Logit-level indices: Position in flattened endogenous tensor (accounting for cardinality)
    
    These precomputed indices avoid repeated computation during training.
    
    Args:
        annotations: Concept annotations with type and cardinality metadata
        
    Returns:
        Dict[str, list]: Dictionary with the following keys:
            - 'binary_labels': List of binary concept names
            - 'categorical_labels': List of categorical concept names
            - 'continuous_labels': List of continuous concept names
            - 'binary_idx': List of concept-level indices for binary concepts
            - 'categorical_idx': List of concept-level indices for categorical concepts
            - 'continuous_idx': List of concept-level indices for continuous concepts
            - 'binary_endogenous_idx': List of logit-level indices for binary concepts
            - 'categorical_endogenous_idx': List of logit-level indices for categorical concepts
            - 'continuous_endogenous_idx': List of logit-level indices for continuous concepts
            
    Example:
        >>> groups = get_concept_groups(annotations)
        >>> binary_endogenous = endogenous[:, groups['binary_endogenous_idx']]  # Extract endogenous of binary concepts
        >>> binary_labels = concept_labels[:, groups['binary_idx']]  # Extract labels of binary concepts
    """
    cardinalities = annotations.cardinalities
    
    # Group concepts by type
    type_groups = annotations.groupby_metadata('type', layout='labels')

    # Concept-level labels: label names
    discrete_labels = type_groups.get('discrete', [])
    continuous_labels = type_groups.get('continuous', [])
    
    # Further split discrete into binary and categorical
    binary_labels = [label for label in discrete_labels if cardinalities[annotations.get_index(label)] == 1]
    categorical_labels = [label for label in discrete_labels if cardinalities[annotations.get_index(label)] > 1]

    # Concept-level indices: position in concept list
    discrete_idx = [annotations.get_index(label) for label in discrete_labels]
    continuous_idx = [annotations.get_index(label) for label in continuous_labels]
    binary_idx = [annotations.get_index(label) for label in binary_labels]
    categorical_idx = [annotations.get_index(label) for label in categorical_labels]

    # Pre-compute cumulative indices for endogenous-level(e.g., logits-level (endogenous) slicing
    # cumulative_indices[i] gives the starting position of concept i in the flattened tensor
    # cumulative_indices[i+1] gives the ending position (exclusive)
    cum_idx = [0] + list(torch.cumsum(torch.tensor(cardinalities), dim=0).tolist())

    # Endogenous (logit-level) indices: position in flattened tensor (accounting for cardinality)
    # These are the actual indices in the output tensor where each concept's logits appear
    binary_endogenous_idx = []
    for concept_idx in binary_idx:
        binary_endogenous_idx.extend(range(cum_idx[concept_idx], cum_idx[concept_idx + 1]))
    
    categorical_endogenous_idx = []
    for concept_idx in categorical_idx:
        categorical_endogenous_idx.extend(range(cum_idx[concept_idx], cum_idx[concept_idx + 1]))
    
    continuous_endogenous_idx = []
    for concept_idx in continuous_idx:
        continuous_endogenous_idx.extend(range(cum_idx[concept_idx], cum_idx[concept_idx + 1]))
    
    return {
        'binary_labels': binary_labels,
        'categorical_labels': categorical_labels,
        'continuous_labels': continuous_labels,
        'binary_idx': binary_idx,
        'categorical_idx': categorical_idx,
        'continuous_idx': continuous_idx,
        'binary_endogenous_idx': binary_endogenous_idx,
        'categorical_endogenous_idx': categorical_endogenous_idx,
        'continuous_endogenous_idx': continuous_endogenous_idx,
    }


def indices_to_mask(
    c_idxs: Union[List[int], torch.Tensor],
    c_vals: Union[List[float], torch.Tensor],
    n_concepts: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert index-based interventions to mask-based format.

    This helper translates interventions specified as (indices, values) pairs
    into (mask, target) tensors, enabling uniform "mask-space" processing while
    supporting intuitive index-based specifications for inference/practice.

    Args:
        c_idxs: Concept indices to intervene on. Can be a list or tensor of shape [K].
        c_vals: Intervention values for each concept. Can be a list or tensor of shape [K]
            (same value for all batches) or [B, K] (per-batch values).
        n_concepts: Total number of concepts (F).
        batch_size: Batch size (B). Default: 1.
        device: Target device for output tensors. Default: None (CPU).
        dtype: Target dtype for output tensors. Default: None (float32).

    Returns:
        tuple: (mask, target) where:
            - mask: Binary tensor of shape [B, F] where 0 indicates intervention, 1 keeps prediction.
            - target: Target tensor of shape [B, F] with intervention values at specified indices.
              Non-intervened positions are set to 0.0 (arbitrary, as they're masked out).

    Example:
        >>> from torch_concepts.nn import indices_to_mask
        >>> # Intervene on concepts 0 and 2, setting them to 1.0 and 0.5
        >>> mask, target = indices_to_mask(
        ...     c_idxs=[0, 2],
        ...     c_vals=[1.0, 0.5],
        ...     n_concepts=5,
        ...     batch_size=2
        ... )
        >>> print(mask.shape, target.shape)
        torch.Size([2, 5]) torch.Size([2, 5])
        >>> print(mask[0])  # [0, 1, 0, 1, 1] - intervene on 0 and 2
        tensor([0., 1., 0., 1., 1.])
        >>> print(target[0])  # [1.0, 0, 0.5, 0, 0]
        tensor([1.0000, 0.0000, 0.5000, 0.0000, 0.0000])
    """
    if dtype is None:
        dtype = torch.float32

    # Convert indices to tensor
    if not isinstance(c_idxs, torch.Tensor):
        c_idxs = torch.tensor(c_idxs, dtype=torch.long, device=device)
    else:
        c_idxs = c_idxs.to(dtype=torch.long, device=device)

    # Convert values to tensor
    if not isinstance(c_vals, torch.Tensor):
        c_vals = torch.tensor(c_vals, dtype=dtype, device=device)
    else:
        c_vals = c_vals.to(dtype=dtype, device=device)

    # Validate indices
    K = c_idxs.numel()
    if K == 0:
        # No interventions - return all-ones mask and zeros target
        mask = torch.ones((batch_size, n_concepts), dtype=dtype, device=device)
        target = torch.zeros((batch_size, n_concepts), dtype=dtype, device=device)
        return mask, target

    if c_idxs.dim() != 1:
        raise ValueError(f"c_idxs must be 1-D, got shape {c_idxs.shape}")

    if torch.any(c_idxs < 0) or torch.any(c_idxs >= n_concepts):
        raise ValueError(f"All indices must be in range [0, {n_concepts}), got {c_idxs}")

    # Handle c_vals shape: [K] or [B, K]
    if c_vals.dim() == 1:
        if c_vals.numel() != K:
            raise ValueError(f"c_vals length {c_vals.numel()} must match c_idxs length {K}")
        # Broadcast to [B, K]
        c_vals = c_vals.unsqueeze(0).expand(batch_size, -1)
    elif c_vals.dim() == 2:
        B_vals, K_vals = c_vals.shape
        if K_vals != K:
            raise ValueError(f"c_vals second dim {K_vals} must match c_idxs length {K}")
        if B_vals != batch_size:
            raise ValueError(f"c_vals first dim {B_vals} must match batch_size {batch_size}")
    else:
        raise ValueError(f"c_vals must be 1-D or 2-D, got shape {c_vals.shape}")

    # Initialize mask (1 = keep prediction, 0 = replace with target)
    mask = torch.ones((batch_size, n_concepts), dtype=dtype, device=device)

    # Initialize target (arbitrary values for non-intervened positions)
    target = torch.zeros((batch_size, n_concepts), dtype=dtype, device=device)

    # Set mask to 0 at intervention indices
    mask[:, c_idxs] = 0.0

    # Set target values at intervention indices
    target[:, c_idxs] = c_vals

    return mask, target
