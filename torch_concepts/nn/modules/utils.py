from typing import Mapping, Optional, Tuple, Dict
import warnings
import torch

from torch_concepts import AxisAnnotation

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


def get_concept_groups(annotations: AxisAnnotation) -> Dict[str, list]:
    """Compute concept grouping by type for efficient loss/metric computation.
    
    Creates index mappings to slice tensors by concept type. Returns indices at two levels:
    1. Concept-level indices: Position in concept list (e.g., concept 0, 1, 2...)
    2. Logit-level indices: Position in flattened logits tensor (accounting for cardinality)
    
    These precomputed indices avoid repeated computation during training.
    
    Args:
        annotations: Concept annotations with type and cardinality metadata
        
    Returns:
        Dict with 6 keys:
            - 'binary_concepts': Indices of binary concepts in concept list
            - 'categorical_concepts': Indices of categorical concepts in concept list  
            - 'continuous_concepts': Indices of continuous concepts in concept list
            - 'binary_logits': Indices in flattened logits tensor for binary concepts
            - 'categorical_logits': Indices in flattened logits tensor for categorical concepts
            - 'continuous_logits': Indices in flattened logits tensor for continuous concepts
            
    Example:
        >>> groups = get_concept_groups(annotations)
        >>> binary_logits = logits[:, groups['binary_logits']]  # Extract logits of binary concepts
        >>> binary_labels = concept_labels[:, groups['binary_concepts']]  # Extract labels of binary concepts
    """
    cardinalities = annotations.cardinalities
    
    # Group concepts by type
    type_groups = annotations.groupby_metadata('type', layout='indices')

    # Concept-level indices: position in concept list
    discrete_concepts = type_groups.get('discrete', [])
    binary_concepts = [idx for idx in discrete_concepts if cardinalities[idx] == 1]
    categorical_concepts = [idx for idx in discrete_concepts if cardinalities[idx] > 1]
    continuous_concepts = type_groups.get('continuous', [])

    # Pre-compute cumulative indices for logit-level slicing
    cumulative_indices = [0] + list(torch.cumsum(torch.tensor(cardinalities), dim=0).tolist())

    # Logit-level indices: position in flattened tensor (accounting for cardinality)
    binary_logits = []
    for concept_idx in binary_concepts:
        binary_logits.extend(range(cumulative_indices[concept_idx], cumulative_indices[concept_idx + 1]))
    
    categorical_logits = []
    for concept_idx in categorical_concepts:
        categorical_logits.extend(range(cumulative_indices[concept_idx], cumulative_indices[concept_idx + 1]))
    
    continuous_logits = []
    for concept_idx in continuous_concepts:
        continuous_logits.extend(range(cumulative_indices[concept_idx], cumulative_indices[concept_idx + 1]))
    
    return {
        'cumulative_indices': cumulative_indices,
        'binary_concepts': binary_concepts,
        'categorical_concepts': categorical_concepts,
        'continuous_concepts': continuous_concepts,
        'binary_logits': binary_logits,
        'categorical_logits': categorical_logits,
        'continuous_logits': continuous_logits,
    }