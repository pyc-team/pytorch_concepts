from typing import Optional, Dict, Union, List, Any
import warnings
import logging
import torch

from ...annotations import Annotations

logger = logging.getLogger(__name__)

#: The concept types a loss or a metric collection routes over.
TYPES = ("binary", "categorical", "continuous")


def by_type(binary=None, categorical=None, continuous=None) -> Dict[str, Any]:
    """``{concept type: item}`` for the types the caller actually gave.

    The ``binary``/``categorical``/``continuous`` argument triple is the public
    shape of both :class:`~torch_concepts.nn.ConceptLoss` and
    :class:`~torch_concepts.nn.ConceptMetrics`; this is the plain dict they route
    with internally.
    """
    return {
        name: item
        for name, item in zip(TYPES, (binary, categorical, continuous))
        if item is not None
    }


def check_collection(annotations: Annotations,
                     collection: Dict[str, Any],
                     collection_name: str) -> Dict[str, Any]:
    """Validate loss/metric configurations against concept annotations.
    
    Ensures that:
    1. Required losses/metrics are present for each concept type
    2. Annotation structure (nested vs dense) matches concept types
    3. Unused configurations are warned about
    
    Args:
        annotations (Annotations): Concept annotations with metadata.
        collection (dict): ``{concept type: item}``, as built by :func:`by_type`.
        collection_name (str): Either 'loss' or 'metrics' for error messages.
        
    Returns:
        dict: The configuration, restricted to the concept types that exist.
            
    Raises:
        ValueError: If validation fails (missing required configs, 
            incompatible annotation structure).
            
    Example:
        >>> from torch_concepts.nn.modules.utils import by_type, check_collection
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> from torch_concepts import Annotations
        >>> loss_config = by_type(
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss()
        ... )
        >>> concept_annotations = Annotations(
        ...     labels=['c1', 'c2', 'c3'],
        ...     cardinalities=[1, 3, 2],
        ...     types=['binary', 'categorical', 'categorical'],
        ... )
        >>> filtered_config = check_collection(
        ...     concept_annotations,
        ...     loss_config, 
        ...     'loss'
        ... )
    """
    assert collection_name in ['loss', 'metrics'], (
        f"collection_name must be 'loss' or 'metrics', got '{collection_name}'"
    )

    # Use cached type_groups from Annotations
    groups = annotations.type_groups
    
    has_binary = len(groups['binary']['labels']) > 0
    has_categorical = len(groups['categorical']['labels']) > 0
    has_continuous = len(groups['continuous']['labels']) > 0

    # Extract items from collection
    binary = collection.get('binary')
    categorical = collection.get('categorical')
    continuous = collection.get('continuous')
    
    # Validation rules
    errors = []
    
    # Check nested/dense compatibility
    all_binary = has_binary and not has_categorical and not has_continuous
    all_categorical = has_categorical and not has_binary and not has_continuous
    all_continuous = has_continuous and not has_binary and not has_categorical
    
    if all_binary:
        if annotations.is_nested:
            errors.append("Annotations for all-binary concepts should NOT be nested.")
    elif all_categorical:
        if not annotations.is_nested:
            errors.append("Annotations for all-categorical concepts should be nested.")
    elif all_continuous:
        if annotations.is_nested:
            errors.append("Annotations for all-continuous concepts should NOT be nested.")
    elif has_binary or has_categorical:
        if not annotations.is_nested:
            errors.append("Annotations for mixed concepts should be nested.")
    
    # Check required items are present
    if has_binary and binary is None:
        errors.append(f"{collection_name} missing 'binary' for binary concepts.")
    if has_categorical and categorical is None:
        errors.append(f"{collection_name} missing 'categorical' for categorical concepts.")
    if has_continuous and continuous is None:
        errors.append(f"{collection_name} missing 'continuous' for continuous concepts.")
    
    if errors:
        raise ValueError(f"{collection_name} validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    # # Warnings for unused items
    # if not has_binary and binary is not None:
    #     warnings.warn(f"Binary {collection_name} will be ignored (no binary concepts).")
    # if not has_categorical and categorical is not None:
    #     warnings.warn(f"Categorical {collection_name} will be ignored (no categorical concepts).")
    # if not has_continuous and continuous is not None:
    #     warnings.warn(f"Continuous {collection_name} will be ignored (no continuous concepts).")
    
    # Keep only the types that exist in the data
    filtered = {}
    if has_binary:
        filtered['binary'] = binary
    if has_categorical:
        filtered['categorical'] = categorical
    if has_continuous:
        filtered['continuous'] = continuous
    
    return filtered


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
        >>> from torch_concepts.nn.modules.utils import indices_to_mask
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



# =============================================================================
# Training mode utilities (reusable for other models)
# =============================================================================

from .high.base.learner import BaseLearner

_CLASS_CACHE = {}


def with_training_mode(cls, lightning: bool = False):
    """Create a combined class with BaseLearner mixin for Lightning training.
    
    This utility adds the BaseLearner mixin to any model class when
    lightning=True.  The BaseLearner provides the Lightning training
    loop.  Inference engine selection (train vs eval) is handled
    automatically by the ``inference`` property on BaseModel, which
    returns ``train_inference`` or ``eval_inference`` depending on
    the module's train/eval mode (toggled by ``.train()``/``.eval()``).
    
    Parameters
    ----------
    cls : type
        The base model class.
    lightning : bool, default False
        If True, adds BaseLearner mixin for Lightning training.
        If False, returns the original class (pure PyTorch module).
    
    Returns
    -------
    type
        Combined class with BaseLearner mixin, or original class if lightning is False.
    """
    # No training mode = pure PyTorch module (no learner mixin)
    if not lightning:
        return cls
    
    # Add BaseLearner mixin for Lightning training
    cache_key = (cls, True)
    if cache_key not in _CLASS_CACHE:
        _CLASS_CACHE[cache_key] = type(
            f'{cls.__name__}_Module',
            (cls, BaseLearner),
            {}
        )
    return _CLASS_CACHE[cache_key]