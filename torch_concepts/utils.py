"""
Utility functions for the torch_concepts package.

This module provides various utility functions for working with concept-based models,
including concept name validation, output size computation, explanation analysis,
seeding for reproducibility, and numerical stability checks.
"""
import importlib
import os
from collections import Counter
from copy import deepcopy
from typing import Dict, Union, List, Mapping, Optional
import torch, math
import logging
from pytorch_lightning import seed_everything as pl_seed_everything

from .annotations import Annotations, AxisAnnotation
from .nn.modules.utils import GroupConfig


def resolve_hf_token() -> Optional[str]:
    """Resolve an HF token from env vars or conceptarium.env fallback.

    Priority order:
    1. HF_TOKEN
    2. HUGGINGFACE_HUB_TOKEN
    3. HUGGINGFACEHUB_TOKEN
    4. conceptarium.env.HUGGINGFACEHUB_TOKEN (if importable)
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_TOKEN")
    )
    if token:
        return token

    try:
        from conceptarium.env import HUGGINGFACEHUB_TOKEN as config_token
    except Exception:
        config_token = None

    if config_token:
        os.environ.setdefault("HF_TOKEN", config_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", config_token)
        return config_token

    return None


def seed_everything(seed: int, workers: bool = True) -> int:
    """Set random seeds across all libraries for reproducibility.
    
    Enhanced wrapper around PyTorch Lightning's seed_everything that also sets
    PYTHONHASHSEED environment variable for complete reproducibility, including
    Python's hash randomization.
    
    Sets seeds for:
    - Python's random module
    - NumPy's random module  
    - PyTorch (CPU and CUDA)
    - PYTHONHASHSEED environment variable
    - PL_GLOBAL_SEED environment variable (via Lightning)
    
    Args:
        seed: Random seed value to set across all libraries.
        workers: If True, sets worker seed for DataLoaders.
        
    Returns:
        The seed value that was set.
        
    Example:
        >>> import torch_concepts as tc
        >>> tc.seed_everything(42)
        42
        >>> # All random operations are now reproducible
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    return pl_seed_everything(seed, workers=workers)


def validate_and_generate_concept_names(
    concept_names: Dict[int, Union[int, List[str]]],
) -> Dict[int, List[str]]:
    """
    Validate and generate concept names based on the provided dictionary.

    Args:
        concept_names: Dictionary where keys are dimension indices and values
            are either integers (indicating the size of the dimension) or lists
            of strings (concept names).

    Returns:
        Dict[int, List[str]]: Processed dictionary with concept names.
    """
    processed_concept_names = {}
    for dim, value in concept_names.items():
        if dim == 0:
            # Batch size dimension is expected to be empty
            processed_concept_names[dim] = []
        elif isinstance(value, int):
            processed_concept_names[dim] = [f"concept_{dim}_{i}" for i in range(value)]
        elif isinstance(value, list):
            processed_concept_names[dim] = value
        else:
            raise ValueError(
                f"Invalid value for dimension {dim}: must be either int or "
                "list of strings."
            )
    return processed_concept_names


def compute_output_size(concept_names: Dict[int, Union[int, List[str]]]) -> int:
    """
    Compute the output size of the linear layer based on the concept names.

    Args:
        concept_names: Dictionary where keys are dimension indices and values
            are either integers (indicating the size of the dimension) or lists
            of strings (concept names).

    Returns:
        int: Computed output size.
    """
    output_size = 1
    for dim, value in concept_names.items():
        if dim != 0:  # Skip batch size dimension
            if isinstance(value, int):
                output_size *= value
            elif isinstance(value, list):
                output_size *= len(value)
    return output_size


def get_most_common_expl(
    explanations: List[Dict[str, str]], n=10
) -> Dict[str, Dict[str, int]]:
    """
    Get the most common explanations for each class. This function receives a
    list of explanations and returns the most common explanations for each
    class. The list of explanations is expected to be a list of dictionaries
    containing the explanations for each sample. The value of the key
    should be the explanation string. Each dictionary (sample) may contain a
    single or multiple explanations for different classes.
    Args:
        explanations: List of explanations
        n: Number of most common explanations to return

    Returns:
        Dict[str, Dict[str, int]]: Dictionary with the most common
            explanations for each class.
    """
    exp_per_class = {}
    for exp in explanations:
        for class_, explanation in exp.items():
            if class_ not in exp_per_class:
                exp_per_class[class_] = []
            exp_per_class[class_].append(explanation)

    most_common_expl = {}

    for class_, explanations in exp_per_class.items():
        most_common_expl[class_] = dict(Counter(explanations).most_common(n))

    return most_common_expl


def compute_temperature(epoch, num_epochs):
    """
    Compute temperature for annealing schedules.

    Computes a temperature value that exponentially decreases from an initial
    temperature of 1.0 to a final temperature of 0.5 over the course of training.

    Args:
        epoch (int): Current training epoch.
        num_epochs (int): Total number of training epochs.

    Returns:
        torch.Tensor: The computed temperature value for the current epoch.
    """
    final_temp = torch.tensor([0.5])
    init_temp = torch.tensor([1.0])
    rate = (math.log(final_temp) - math.log(init_temp)) / float(num_epochs)
    curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
    return curr_temp


def numerical_stability_check(cov, device, epsilon=1e-6):
    """
    Check for numerical stability of covariance matrix.
    If not stable (i.e., not positive definite), add epsilon to diagonal.

    Parameters:
    cov (Tensor): The covariance matrix to check.
    epsilon (float, optional): The value to add to the diagonal if the matrix is not positive definite. Default is 1e-6.

    Returns:
    Tensor: The potentially adjusted covariance matrix.
    """
    num_added = 0
    if cov.dim() == 2:
        cov = (cov + cov.transpose(dim0=0, dim1=1)) / 2
    else:
        cov = (cov + cov.transpose(dim0=1, dim1=2)) / 2

    while True:
        try:
            # Attempt Cholesky decomposition; if it fails, the matrix is not positive definite
            torch.linalg.cholesky(cov)
            if num_added > 0.0001:
                logging.warning(f"Added {num_added} to the diagonal of the covariance matrix.")
            break
        except RuntimeError:
            # Add epsilon to the diagonal
            if cov.dim() == 2:
                cov = cov + epsilon * torch.eye(cov.size(0), device=device)
            else:
                cov = cov + epsilon * torch.eye(cov.size(1), device=device)
            num_added += epsilon
            epsilon *= 2
    return cov


def _is_int_index(x) -> bool:
    """
    Check if a value is an integer index.

    Args:
        x: Value to check.

    Returns:
        bool: True if x is an int or 0-dimensional tensor, False otherwise.
    """
    return isinstance(x, int) or (isinstance(x, torch.Tensor) and x.dim() == 0)


def _check_tensors(tensors):
    """
    Validate that a list of tensors are compatible for concatenation.

    Ensures all tensors have:
    - At least 2 dimensions (batch and concept dimensions)
    - Same batch size (dimension 0)
    - Same trailing dimensions (dimension 2+)
    - Same dtype and device
    - Same requires_grad setting

    The concept dimension (dimension 1) is allowed to vary.

    Args:
        tensors (List[torch.Tensor]): List of tensors to validate.

    Raises:
        ValueError: If tensors have incompatible shapes, dtypes, devices, or settings.
    """
    # First, check that all tensors have at least 2 dimensions
    for i, t in enumerate(tensors):
        if t.dim() < 2:
            raise ValueError(f"Tensor {i} must have at least 2 dims (B, c_i, ...); got {tuple(t.shape)}.")

    # Check that all tensors have the same number of dimensions
    first_ndim = tensors[0].dim()
    for i, t in enumerate(tensors):
        if t.dim() != first_ndim:
            raise ValueError(f"All tensors must have at least 2 dims and the same total number of dimensions; Tensor 0 has {first_ndim} dims, but Tensor {i} has {t.dim()} dims.")

    B = tensors[0].shape[0]
    dtype = tensors[0].dtype
    device = tensors[0].device
    rest_shape = tensors[0].shape[2:]  # dims >=2 must match

    for i, t in enumerate(tensors):
        if t.shape[0] != B:
            raise ValueError(f"All tensors must share batch dim. Got {t.shape[0]} != {B} at field {i}.")
        # only dim=1 may vary; dims >=2 must match exactly
        if t.shape[2:] != rest_shape:
            raise ValueError(
                f"All tensors must share trailing shape from dim=2. "
                f"Field {i} has {t.shape[2:]} != {rest_shape}."
            )
        if t.dtype != dtype:
            raise ValueError("All tensors must share dtype.")
        if t.device != device:
            raise ValueError("All tensors must be on the same device.")
        if t.requires_grad != tensors[0].requires_grad:
            raise ValueError("All tensors must have the same requires_grad setting.")


def _get_type_group(metadata: Dict, cardinality: int) -> str:
    """Classify a concept into 'binary', 'categorical', or 'continuous' based on metadata type and cardinality.
    
    Args:
        metadata: Per-concept metadata dict (must contain 'type' key).
        cardinality: Cardinality of the concept.

    Returns:
        One of 'binary', 'categorical', or 'continuous'.

    Raises:
        ValueError: If the combination of type and cardinality is not recognized.
    """
    concept_type = metadata.get('type', 'discrete')
    if concept_type == 'discrete' and cardinality == 1:
        return 'binary'
    elif concept_type == 'discrete' and cardinality > 1:
        return 'categorical'
    elif concept_type == 'continuous':
        return 'continuous'
    else:
        raise ValueError(f"Unrecognized type/cardinality combination: type={concept_type!r}, cardinality={cardinality}")


def add_property_to_annotations(
        annotations: Union[Annotations, AxisAnnotation],
        values: Union[GroupConfig, Mapping[str, object]],
        property_name: str,
    ) -> Union[Annotations, AxisAnnotation]:
    """Add a metadata property (e.g. 'distribution' or 'activation') to annotations.

    Accepts either a ``GroupConfig`` (keyed by type group: binary/categorical/continuous)
    or a ``Mapping`` (keyed by concept name).
    Accepts either a ``GroupConfig`` (keyed by type group: binary/categorical/continuous)
    or a ``Mapping`` (keyed by concept name).

    Args:
        annotations: Annotations or AxisAnnotation to update.
        values: A ``GroupConfig`` or ``Mapping`` providing the property value per concept.
        property_name: Metadata key to write (e.g. ``'distribution'``, ``'activation'``).

    Returns:
        Updated annotations with the property added to each concept's metadata.
    """
    if isinstance(annotations, Annotations):
        axis_annotation = annotations.get_axis_annotation(1)
    elif isinstance(annotations, AxisAnnotation):
        axis_annotation = annotations
    else:
        raise ValueError("annotations must be either Annotations or AxisAnnotation instance.")

    new_metadata = deepcopy(axis_annotation.metadata)
    cardinalities = axis_annotation.cardinalities

    if isinstance(values, GroupConfig):
        for (concept_name, metadata), cardinality in zip(axis_annotation.metadata.items(), cardinalities):
            group = _get_type_group(metadata, cardinality)
            try:
                entry = values[group]
            except KeyError:
                raise ValueError(
                    f"No {property_name} config for type group '{group}' "
                    f"(concept '{concept_name}'). "
                )

            # entry is either a bare class/value or [class, {kwargs}]
            kwargs_key = 'dist_kwargs' if property_name == 'distribution' else f'{property_name}_kwargs'
            if isinstance(entry, (list, tuple)):
                new_metadata[concept_name][property_name] = entry[0]
                if len(entry) > 1 and isinstance(entry[1], dict):
                    new_metadata[concept_name][kwargs_key] = dict(entry[1])
            else:
                new_metadata[concept_name][property_name] = entry
    elif isinstance(values, Mapping):
        for concept_name in axis_annotation.metadata.keys():
            value = values.get(concept_name, None)
            if value is None:
                raise ValueError(f"No {property_name} config found for concept '{concept_name}'.")
            new_metadata[concept_name][property_name] = value
    else:
        raise ValueError(f"{property_name} must be a GroupConfig or a Mapping.")

    axis_annotation.metadata = new_metadata
    if isinstance(annotations, Annotations):
        annotations[1] = axis_annotation
        return annotations
    else:
        return axis_annotation


def add_distribution_to_annotations(
        annotations: Union[Annotations, AxisAnnotation],
        distributions: Union[GroupConfig, Mapping[str, object]],
    ) -> Union[Annotations, AxisAnnotation]:
    """Add distribution classes to annotation metadata.

    Args:
        annotations: Annotations or AxisAnnotation to update.
        distributions: Distribution classes per concept. Either a
            ``GroupConfig`` (keyed by type group: ``'binary'``, ``'categorical'``,
            ``'continuous'``) or a ``Mapping`` from concept names to distribution
            classes.

    Returns:
        Updated annotations with ``'distribution'`` added to each concept's metadata.

    Example:
        >>> from torch.distributions import Bernoulli, OneHotCategorical
        >>> from torch_concepts import GroupConfig
        >>> distributions = GroupConfig(binary=Bernoulli, categorical=OneHotCategorical)
        >>> annotations = add_distribution_to_annotations(annotations, distributions)
    """
    return add_property_to_annotations(annotations, distributions, 'distribution')


def add_activation_to_annotations(
        annotations: Union[Annotations, AxisAnnotation],
        activations: Union[GroupConfig, Mapping[str, object]],
    ) -> Union[Annotations, AxisAnnotation]:
    """Add activation functions to annotation metadata.

    Args:
        annotations: Annotations or AxisAnnotation to update.
        activations: Activation functions per concept. Either a
            ``GroupConfig`` (keyed by type group: ``'binary'``, ``'categorical'``,
            ``'continuous'``) or a ``Mapping`` from concept names to callables.

    Returns:
        Updated annotations with ``'activation'`` added to each concept's metadata.

    Example:
        >>> from functools import partial
        >>> from torch_concepts import GroupConfig
        >>> activations = GroupConfig(binary=torch.sigmoid, categorical=partial(torch.softmax, dim=-1))
        >>> annotations = add_activation_to_annotations(annotations, activations)
    """
    return add_property_to_annotations(annotations, activations, 'activation')


def add_default_properties(
        annotations: Union[Annotations, AxisAnnotation],
    ) -> Union[Annotations, AxisAnnotation]:
    """Fill in default distributions and activations for concepts that lack them.

    For each concept missing a ``'distribution'``, assigns a default based on
    the concept's type group (binary/categorical/continuous) using
    :data:`~torch_concepts.nn.modules.mid.models.variable._DEFAULT_DISTRIBUTIONS`.

    For each concept missing an ``'activation'`` (but having a ``'distribution'``),
    assigns a default using
    :data:`~torch_concepts.nn.modules.mid.models.variable._DEFAULT_ACTIVATIONS`.

    If no default can be determined for a concept, a ``ValueError`` is raised
    asking the user to add the property to the annotation object explicitly
    (e.g. via :func:`add_property_to_annotations`).

    Args:
        annotations: Annotations or AxisAnnotation with per-concept metadata.

    Returns:
        Updated annotations with defaults filled in.

    Raises:
        ValueError: If a concept is missing a distribution or activation and
            no default exists for its type group / distribution class.
    """
    from .nn.modules.mid.models.variable import _DEFAULT_DISTRIBUTIONS, _DEFAULT_ACTIVATIONS

    if isinstance(annotations, Annotations):
        axis_annotation = annotations.get_axis_annotation(1)
    elif isinstance(annotations, AxisAnnotation):
        axis_annotation = annotations
    else:
        raise ValueError("annotations must be either Annotations or AxisAnnotation instance.")

    cardinalities = axis_annotation.cardinalities

    # --- Default distributions ---
    for (label, metadata), cardinality in zip(axis_annotation.metadata.items(), cardinalities):
        if 'distribution' not in metadata:
            group = _get_type_group(metadata, cardinality)
            if group in _DEFAULT_DISTRIBUTIONS:
                metadata['distribution'] = _DEFAULT_DISTRIBUTIONS[group]
            else:
                raise ValueError(
                    f"No default distribution for type group '{group}' "
                    f"(concept '{label}'). Please add a 'distribution' to the "
                    f"annotation metadata, e.g. via "
                    f"add_property_to_annotations()."
                )

    # --- Default activations ---
    for label in axis_annotation.labels:
        meta = axis_annotation.metadata[label]
        if 'activation' not in meta:
            dist = meta.get('distribution')
            if dist is not None and dist in _DEFAULT_ACTIVATIONS:
                meta['activation'] = _DEFAULT_ACTIVATIONS[dist]
            elif dist is not None:
                raise ValueError(
                    f"No default activation for distribution "
                    f"{dist.__name__} of concept '{label}'. "
                    f"Please add an 'activation' to the annotation metadata, "
                    f"e.g. via add_property_to_annotations()."
                )

    if isinstance(annotations, Annotations):
        annotations[1] = axis_annotation
        return annotations
    else:
        return axis_annotation


def get_from_string(class_path: str):
    """Import and return a class from its fully qualified string path.

    Args:
        class_path: Fully qualified class path (e.g., 'torch.optim.Adam').

    Returns:
        Class object (not instantiated).

    Example:
        >>> Adam = get_from_string('torch.optim.Adam')
        >>> optimizer = Adam(model.parameters(), lr=0.001)
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def instantiate_from_string(class_path: str, **kwargs):
    """Instantiate a class from its fully qualified string path.
    
    Args:
        class_path: Fully qualified class path (e.g., 'torch.nn.ReLU').
        **kwargs: Keyword arguments passed to class constructor.
        
    Returns:
        Instantiated class object.
        
    Example:
        >>> relu = instantiate_from_string('torch.nn.ReLU')
        >>> loss = instantiate_from_string(
        ...     'torch.nn.BCEWithLogitsLoss', reduction='mean'
        ... )
    """
    cls = get_from_string(class_path)
    return cls(**kwargs)
