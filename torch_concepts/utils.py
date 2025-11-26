"""
Utility functions for the torch_concepts package.

This module provides various utility functions for working with concept-based models,
including concept name validation, output size computation, explanation analysis,
seeding for reproducibility, and numerical stability checks.
"""
import importlib
import os
import warnings
from collections import Counter
from copy import deepcopy
from typing import Dict, Union, List, Mapping
import torch, math
import logging
from pytorch_lightning import seed_everything as pl_seed_everything

from .annotations import Annotations, AxisAnnotation
from .nn.modules.utils import GroupConfig


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


def add_distribution_to_annotations(
        annotations: Union[Annotations, AxisAnnotation],
        distributions: Union[GroupConfig, Mapping[str, object]]
    ) -> Union[Annotations, AxisAnnotation]:
    """
    Add probability distribution classes to concept annotations metadata.

    This function updates the metadata of each concept in the provided AxisAnnotation
    by assigning a probability distribution class/config based on the concept's type
    ('discrete' or 'continuous') and cardinality. The distribution can be provided
    either as a GroupConfig (with keys 'binary' / 'categorical' / 'continuous') or as a Mapping
    from concept names to distributions.

    Args:
        annotations (AxisAnnotation): Concept annotations containing metadata and cardinalities.
        distributions (GroupConfig or Mapping): Either a GroupConfig with keys
            'binary' / 'categorical' / 'continuous', or a Mapping from concept names to distributions.

    Returns:
        AxisAnnotation: Updated annotations with a 'distribution' field added to each concept's metadata.

    Example:
        >>> from torch_concepts.annotations import AxisAnnotation
        >>> from torch_concepts.nn.modules.utils import GroupConfig
        >>> annotations = AxisAnnotation(
        ...     metadata={
        ...         'color': {'type': 'discrete'},
        ...         'size': {'type': 'discrete'},
        ...     },
        ...     cardinalities=[3, 1]
        ... )
        >>> distributions = GroupConfig(
        ...     binary = torch.distributions.Bernoulli(),
        ...     categorical = torch.distributions.Categorical()
        ... )
        >>> updated = add_distribution_to_annotations(annotations, distributions)
        >>> print(updated.metadata['color']['distribution'])
        {'path': 'torch.distributions.Categorical'}
        >>> print(updated.metadata['size']['distribution'])
        {'path': 'torch.distributions.Bernoulli'}
    """
    if isinstance(annotations, Annotations):
        axis_annotation = annotations.get_axis_annotation(1)
    elif isinstance(annotations, AxisAnnotation):
        axis_annotation = annotations
    else:
        raise ValueError("annotations must be either Annotations or AxisAnnotation instance.")
    new_metadata = deepcopy(axis_annotation.metadata)
    cardinalities = axis_annotation.cardinalities

    if isinstance(distributions, GroupConfig):
        for (concept_name, metadata), cardinality in zip(axis_annotation.metadata.items(), cardinalities):
            if metadata['type'] == 'discrete' and cardinality == 1:
                new_metadata[concept_name]['distribution'] = distributions['binary']
            elif metadata['type'] == 'discrete' and cardinality > 1:
                new_metadata[concept_name]['distribution'] = distributions['categorical']
            elif metadata['type'] == 'continuous' and cardinality == 1:
                raise NotImplementedError("Continuous concepts not supported yet.")
            elif metadata['type'] == 'continuous' and cardinality > 1:
                raise NotImplementedError("Continuous concepts not supported yet.")
            else:
                raise ValueError(f"Cannot set distribution type for concept {concept_name}.")
    elif isinstance(distributions, Mapping):
        for concept_name in axis_annotation.metadata.keys():
            dist = distributions.get(concept_name, None)
            if dist is None:
                raise ValueError(f"No distribution config found for concept {concept_name}.")
            new_metadata[concept_name]['distribution'] = dist
    else:
        raise ValueError("Distributions must be a GroupConfig or a Mapping.")
    axis_annotation.metadata = new_metadata
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
