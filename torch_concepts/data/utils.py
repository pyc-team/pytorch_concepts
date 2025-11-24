"""
Data utility functions for tensor manipulation and transformation.

This module provides utility functions for data processing, including tensor
conversion, image colorization, and affine transformations.
"""
import os
import numpy as np
import pandas as pd
import logging
from typing import Any, List, Sequence, Union
import torch
import random
from torch import Tensor
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


def ensure_list(value: Any) -> List:
    """
    Ensure a value is converted to a list. If the value is iterable (but not a 
    string or dict), converts it to a list. Otherwise, wraps it in a list.

    Args:
        value: Any value to convert to list.

    Returns:
        List: The value as a list.
    
    Examples:
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list((1, 2, 3))
        [1, 2, 3]
        >>> ensure_list(5)
        [5]
        >>> ensure_list("hello")
        ['hello']
        >>> ensure_list({'a': 1, 'b': 2})  # doctest: +SKIP
        TypeError: Cannot convert dict to list. Use list(dict.values()) 
        or list(dict.keys()) explicitly.
    """
    # Explicitly reject dictionaries to avoid silent conversion to keys
    if isinstance(value, dict):
        raise TypeError(
            "Cannot convert dict to list. Use list(dict.values()) or " \
            "list(dict.keys()) explicitly to make your intent clear."
        )
    
    # Check for iterables (but not strings)
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    else:
        return [value]

def files_exist(files: Sequence[str]) -> bool:
    """
    Check if all files in a sequence exist.

    Args:
        files: Sequence of file paths to check.

    Returns:
        bool: True if all files exist, False otherwise.
              Returns True for empty sequences (vacuous truth).
    """
    files = ensure_list(files)
    return all([os.path.exists(f) for f in files])

def parse_tensor(data: Union[np.ndarray, pd.DataFrame, Tensor],
                name: str,
                precision: Union[int, str]) -> Tensor:
    """
    Convert input data to torch tensor with appropriate format.

    Supports conversion from numpy arrays, pandas DataFrames, or existing tensors.

    Args:
        data: Input data as numpy array, DataFrame, or Tensor.
        name: Name of the data (for error messages).
        precision: Desired numerical precision (16, 32, or 64).

    Returns:
        Tensor: Converted tensor with specified precision.

    Raises:
        AssertionError: If data is not in a supported format.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values)
    else:
        assert isinstance(data, Tensor), f"{name} must be np.ndarray, \
            pd.DataFrame, or torch.Tensor"
    return convert_precision(data, precision)

def convert_precision(tensor: Tensor,
                       precision: Union[int, str]) -> Tensor:
    """
    Convert tensor to specified precision.

    Args:
        tensor: Input tensor.
        precision: Target precision ("float16", "float32", or "float64", or 16, 32, 64).

    Returns:
        Tensor: Tensor converted to specified precision.
    """
    if precision == "float32":
        tensor = tensor.to(torch.float32)
    elif precision == "float64":
        tensor = tensor.to(torch.float64)
    elif precision == "float16":
        tensor = tensor.to(torch.float16)
    return tensor

def resolve_size(size: Union[int, float], n_samples: int) -> int:
    """Convert size specification to absolute number of samples.
    
    Args:
        size: Either an integer (absolute count) or float (fraction in [0, 1]).
        n_samples: Total number of samples in dataset.
        
    Returns:
        int: Absolute number of samples.
        
    Raises:
        ValueError: If fractional size is not in [0, 1] or absolute size is negative.
        TypeError: If size is neither int nor float.
    """
    if isinstance(size, float):
        if not 0.0 <= size <= 1.0:
            raise ValueError(f"Fractional size must be in [0, 1], got {size}")
        return int(size * n_samples)
    
    elif isinstance(size, int):
        if size < 0:
            raise ValueError(f"Absolute size must be non-negative, got {size}")
        return size
    
    else:
        raise TypeError(f"Size must be int or float, got {type(size).__name__}")
        
def colorize(images, colors):
    """
    Colorize grayscale images based on specified colors.

    Converts grayscale images to RGB by assigning the intensity to one
    of three color channels (red, green, or blue).

    Args:
        images: Tensor of shape (N, H, W) containing grayscale images.
        colors: Tensor of shape (N) containing color labels (0=red, 1=green, 2=blue).

    Returns:
        Tensor: Colored images of shape (N, 3, H, W).

    Raises:
        AssertionError: If colors contain values other than 0, 1, or 2.
    """
    assert torch.unique(colors).shape[0] <= 3, "colors must be 0, 1, or 2 (red, green, blue)."
    N = images.shape[0]
    colored_images = torch.zeros((N, 3, images.shape[1], images.shape[2]), dtype=images.dtype, device=images.device)
    indices = torch.arange(N)
    colored_images[indices, colors, :, :] = images
    return colored_images

def affine_transform(images, degrees, scales, batch_size=512):
    """
    Apply affine transformations to a batch of images.

    Applies rotation and scaling transformations to each image.

    Args:
        images: Tensor of shape (N, H, W) or (N, 3, H, W).
        degrees: Tensor of shape (N) containing rotation degrees.
        scales: Tensor of shape (N) containing scaling factors.
        batch_size: Number of images to process at once (default: 512).

    Returns:
        Tensor: Transformed images with same shape as input.
    """
    if degrees is None:
        logger.warning("Degrees for affine transformation of images not provided, setting to 0.")
        degrees = torch.zeros(images.shape[0], device=images.device)
    if scales is None:
        logger.warning("Scales for affine transformation of images not provided, setting to 1.")
        scales = torch.ones(images.shape[0], device=images.device)

    N = images.shape[0]
    if images.dim() == 3:
        images = images.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)

    for i in range(0, N, batch_size):
        imgs = images[i:i+batch_size]
        degs = degrees[i:i+batch_size]
        scs = scales[i:i+batch_size]

        transformed = torch.stack([
            v2.RandomAffine(degrees=(deg.item(), deg.item()), scale=(sc.item(), sc.item()))(img)
            for img, deg, sc in zip(imgs, degs, scs)
        ])

        images[i:i+batch_size] = transformed

    return images


def transform_images(images, transformations, colors=None, degrees=None, scales=None):
    """
    Apply a sequence of transformations to a batch of images.

    Args:
        images: Tensor of shape [N, H, W] or [N, 3, H, W].
        transformations: List of transformation names (e.g., ['colorize', 'affine']).
        colors: Optional color labels for colorization.
        degrees: Optional rotation degrees for affine transform.
        scales: Optional scaling factors for affine transform.

    Returns:
        Tensor: Transformed images.
    """
    for t in transformations:
        if t == 'colorize':
            if colors is None:
                raise ValueError("Colors must be provided for colorize.")
            images = colorize(images, colors)
        elif t in ['affine']:
            images = affine_transform(images, degrees=degrees, scales=scales)
        else:
            raise ValueError(f"Unknown transformation: {t}")
    return images


def assign_random_values(concept, random_prob=[0.5, 0.5], values = [0,1]):
    """Create a vector of random values for each sample in concepts.
    Args:
        concepts: Tensor of shape (N) containing concept values (e.g. digit labels 0-9).
        random_prob: List of probabilities for each value.
        values: List of output values corresponding to each probability.
    Returns:
        outputs: Tensor of shape (N) containing final values.
    """
    N = len(concept)

    # checks on concept
    assert len(concept.shape) == 1, "concepts must be a 1D tensor."

    # checks on random_prob
    assert len(random_prob) > 0, "random_prob must not be empty."
    assert len(random_prob) == len(values), "random_prob must have the same length as values."
    assert all(0.0 <= p <= 1.0 for p in random_prob), "random_prob must be between 0 and 1."
    assert abs(sum(random_prob) - 1.0) < 1e-6, "random_prob must sum to 1."

    # checks on values
    assert len(values) > 0, "values must not be empty."
    assert len(values) == len(set(values)), "values must be unique."
    
    probs = torch.tensor(random_prob, device=concept.device)
    outputs = torch.multinomial(probs, N, replacement=True)
    outputs_unique = torch.unique(outputs)
    outputs_unique = sorted(outputs_unique)
    mapping = {outputs_unique[i].item(): values[i] for i in range(len(outputs_unique))}
    outputs= torch.tensor([mapping[i.item()] for i in outputs], device=concept.device)

    return outputs

def assign_values_based_on_intervals(concept, intervals, values):
    """Create a vector of values (0 or 1) for each sample in concepts based on intervals given.
    If a concept value belongs to interval[i], it gets an output value randomly chosen among values[i].
    Args:
        concept: Tensor of shape (N) containing concept values (e.g. digit labels 0-9).
        intervals: List of lists, each inner list contains the values defining an interval.
        values: List of lists of output values corresponding to each interval.   
    Returns:
        outputs: Tensor of shape (N) containing final values.
    """
    N = len(concept)

    # checks on ceoncept
    assert len(concept.shape) == 1, "concepts must be a 1D tensor."

    # checks on intervals
    assert len(intervals) == len(values), "intervals and values must have the same length."
    all_interval_values = [item for sublist in intervals for item in sublist]
    assert len(all_interval_values) == len(set(all_interval_values)), "input intervals must not overlap."
    assert all(len(d) > 0 for d in intervals), "each entry in intervals must contain at least one value."

    # checks on values
    assert all(len(v) > 0 for v in values), "each entry in values must contain at least one value."

    outputs = torch.zeros_like(concept)

    # create mask for each interval
    for i, d in enumerate(intervals):
        mask = torch.isin(concept, torch.tensor(d))
        outputs[mask] = i + 1

    # output must be a random value chosen among values[i] for each value i of the mask
    outputs_unique = torch.unique(outputs)
    outputs_unique = sorted(outputs_unique)
    mapping = {outputs_unique[i].item(): values[i] for i in range(len(outputs_unique))}
    outputs = torch.tensor([random.choice(mapping[i.item()]) for i in outputs], device=concept.device)
    return outputs


def colorize_and_transform(data, targets, training_percentage=0.8, test_percentage=0.2, training_mode=['random'], test_mode=['random'], training_kwargs=[{}], test_kwargs=[{}]):
    """Colorize and transform MNIST images based on specified coloring scheme.
       The coloring scheme is defined differently for training and test data.
       It can contain parameters for coloring, scale and rotating images.
    
    Args:
        data: Tensor of shape (N, 28, 28) containing grayscale MNIST images.
        targets: Tensor of shape (N) containing target values (0-9).
        training_percentage: Percentage of data to color for training.
        test_percentage: Percentage of data to color for testing.
        training_mode: List of coloring modes for training data. Options are 'random' and '
        test_mode: List of coloring modes for test data. Options are 'random' and 'digits'.
        training_kwargs: List of dictionaries containing additional arguments for each training mode.
        test_kwargs: List of dictionaries containing additional arguments for each test mode.
 
    Returns:
        input: Tensor of shape (N, 3, 28, 28) containing colorized and/or transformed images.
        concepts: Dictionary containing values of the parameters used for coloring and transformations (e.g., colors, scales, degrees).
        targets: Tensor of shape (N) containing target values (0-9).
        coloring_mode: List of strings indicating the coloring mode used for each sample ('training' or 'test').

    Note: data and targets are shuffled before applying the coloring scheme.
    """
    percentages = {"training": training_percentage, "test": test_percentage}
    mode = {"training": training_mode, "test": test_mode}
    kwargs = {"training": training_kwargs, "test": test_kwargs}
    assert abs(sum(percentages.values()) - 1.0) < 1e-6, "training_percentage and test_percentage must sum to 1."
    

    # check modality, if training_mode or test mode contain "additional_concepts"
    clothing_present = False
    if "additional_concepts_custom" in training_mode or "additional_concepts_custom" in test_mode:
        concepts_used_training = kwargs.get("training", [{}])[0].get("concepts_used", [])
        concepts_used_test = kwargs.get("test", [{}])[0].get("concepts_used", [])
        if "clothing" in kwargs.get("training", [{}])[0].get("concepts_used", []) or "clothing" in kwargs.get("test", [{}])[0].get("concepts_used", []):
            clothing_present = True
            concepts_used_training = [c for c in concepts_used_training if c != "clothing"]
            concepts_used_test = [c for c in concepts_used_test if c != "clothing"]
            assert concepts_used_training == concepts_used_test, "Except for 'clothing', the concepts used must be the same in training and test."
        else:
            assert concepts_used_training == concepts_used_test, "Concepts used must be the same in training and test."


    color_mapping = {'red': 0, 'green': 1, 'blue': 2}

    N = data.shape[0]
    indices = torch.randperm(N)

    embeddings = torch.zeros((N, 3, data.shape[1], data.shape[2]), dtype=data.dtype)
    concepts = {}
    coloring_mode = ["" for _ in range(N)]

    # shuffle data and targets accordingly
    data = data[indices]
    targets = targets[indices]

    start_idx = 0

    for split, perc, m, kw in zip(percentages.keys(), percentages.values(), mode.values(), kwargs.values()):
           
        m = m[0]
        kw = kw[0]
        n_samples = int(perc * N)
        if split == "test":  # last color takes the rest
            end_idx = N
        else:
            end_idx = start_idx + n_samples
        selected_data = data[start_idx:end_idx]
        selected_targets = targets[start_idx:end_idx]
    
        if m == 'random':
            # check keys of kw are exactly the ones expected
            expected_keys = ['random_prob', 'values']
            if set(kw.keys()) != set(expected_keys):
                raise ValueError(f"random coloring requires the following keys in kwargs: {expected_keys}")
            # load values from kw
            prob_mod = kw.get('random_prob')
            colors = kw.get('values')

            # checks on 'random_prob'
            assert isinstance(prob_mod, list), "random_prob must be a list."

            # checks on 'values'
            assert isinstance(colors, list), "values must be a list."
            if not all(v in color_mapping for v in colors):
                raise ValueError(f"All values must be one of {list(color_mapping.keys())}.")
            assert len(colors) == len(set(colors)), "colors must not repeat."

            # transform prob_mod if needed
            if prob_mod[0] == 'uniform':
                random_prob = [1.0 / (len(colors))] * (len(colors))
            else: 
                random_prob = prob_mod
            
            # calculate concept values and transform images accordingly
            numeric_colors = [color_mapping[v] for v in colors]
            random_colors = assign_random_values(selected_targets, random_prob=random_prob, values=numeric_colors)
            colored_data = transform_images(selected_data, transformations=["colorize"], colors=random_colors)
            selected_concepts = {'colors': random_colors}

        elif m == 'intervals':
            # check keys of kw are exactly the ones expected
            expected_keys = ['intervals', 'values']
            if set(kw.keys()) != set(expected_keys):
                raise ValueError(f"intervals coloring requires the following keys in kwargs: {expected_keys}")
            # load values from kw
            interval_values = kw.get('intervals')
            colors = kw.get('values')

            # checks on 'intervals'
            assert all(isinstance(v, list) for v in interval_values), "each entry in intervals must be a list."
            assert len(interval_values) == len(colors), "intervals and values must have the same length."
            all_interval_values = [item for sublist in interval_values for item in sublist]
            unique_targets = torch.unique(selected_targets).tolist()    
            assert set(all_interval_values) == set(unique_targets), f"intervals must cover all target values, i.e.: {unique_targets}"
            assert set(all_interval_values).issubset(set(range(10))), "interval values must be between 0 and 9."
            
            # checks on 'values'
            assert all(isinstance(v, list) for v in colors), "each entry in colors must be a list."
            all_colors_values = [item for sublist in colors for item in sublist]
            if not all(v in color_mapping for v in all_colors_values):
                raise ValueError(f"All values must be one of {list(color_mapping.keys())}.")

            # calculate concept values and transform images accordingly
            numeric_colors = [[color_mapping[v] for v in sublist] for sublist in colors]
            interval_colors = assign_values_based_on_intervals(selected_targets, intervals=interval_values, values=numeric_colors)
            colored_data = transform_images(selected_data, transformations=["colorize"], colors=interval_colors)
            selected_concepts = {'colors': interval_colors}

        elif m == 'additional_concepts_custom':
            # check keys of kw are exactly the ones expected
            expected_keys = ['concepts_used', 'values']
            if set(kw.keys()) != set(expected_keys):
                raise ValueError(f"additional_concepts_custom coloring requires the following keys in kwargs: {expected_keys}")
            # load values from kw
            concepts_used = kw.get('concepts_used')
            values = kw.get('values')

            # checks on 'concepts_used'
            assert isinstance(concepts_used, list), "concepts_used must be a list."
            #assert len(concepts_used) == 3, "There must be 3 concepts used."
            assert len(concepts_used) == len(values), "concepts_used and values must have the same length."
            assert 'colors' in concepts_used, "concepts_used must contain 'color'"

            # checks on 'values'
            assert all(isinstance(v, list) for v in values), "each entry in values must be a list."
            lengths = [len(v) for v in values]
            assert all(l == lengths[0] for l in lengths), "each entry in values must have the same length."
            
            # if "clothing" is in concept_used, check all values are present
            if 'clothing' in concepts_used:
                # it must be in the first position
                assert concepts_used.index('clothing') == 0, "If 'clothing' is used, it must be the first concept."
                clothing_values = values[concepts_used.index('clothing')]
                all_clothing = set(range(10))
                provided_clothing = set([item for sublist in clothing_values for item in sublist])
                assert all_clothing.issubset(provided_clothing), "All clothing values (0-9) must be present in clothing values."
                assert provided_clothing.issubset(all_clothing), "Clothing values must be between 0 and 9."


            # calculate concept values and transform images accordingly
            idx_color = concepts_used.index('colors')
            values[idx_color] = [[color_mapping[c] for c in sublist] for sublist in values[idx_color]]

            if concepts_used[0] !="clothing":
                # if concept 0 is not clothing, assign random values to samples from values[0]
                concept_0_values = [item for sublist in values[0] for item in sublist]
                random_prob = [1.0 / len(concept_0_values)] * (len(concept_0_values))
                concept_0 = assign_random_values(selected_targets, random_prob = random_prob, values = concept_0_values)
            else:
                concept_0 = selected_targets

            selected_concepts = {}
            selected_concepts[concepts_used[0]] = concept_0
            for i in range(1,len(concepts_used)):
                selected_concepts[concepts_used[i]] = assign_values_based_on_intervals(selected_concepts[concepts_used[i-1]], 
                                                                                       intervals = values[i-1], 
                                                                                       values = values[i])

            if 'clothing' in selected_concepts:
                del selected_concepts['clothing']

            idx_scale = concepts_used.index('scales') if 'scales' in concepts_used else None
            idx_degree = concepts_used.index('degrees') if 'degrees' in concepts_used else None
            colored_data = transform_images(selected_data, 
                                            transformations=["colorize", "affine"], 
                                            colors= selected_concepts[concepts_used[idx_color]],
                                            degrees= selected_concepts[concepts_used[idx_degree]] if idx_degree is not None else None, 
                                            scales= selected_concepts[concepts_used[idx_scale]] if idx_scale is not None else None)

        elif m == 'additional_concepts_random':
            # check keys of kw are exactly the ones expected
            expected_keys = ['concepts_used', 'values', 'random_prob']
            if set(kw.keys()) != set(expected_keys):
                raise ValueError(f"additional_concepts_random coloring requires the following keys in kwargs: {expected_keys}")

            # load values from kw
            concepts_used = kw.get('concepts_used', [])
            values = kw.get('values', [])
            prob_mod = kw.get('random_prob')

            # checks on 'concepts_used'
            assert isinstance(concepts_used, list), "concepts_used must be a list."
            assert len(concepts_used) == len(values), "concepts_used and values must have the same length."
            assert len(concepts_used) == len(prob_mod), "concepts_used and random_prob must have the same length."
            assert 'colors' in concepts_used, "concepts_used must contain 'colors'"
            assert 'clothing' not in concepts_used, "'clothing' cannot be used in additional_concepts_random coloring."
            
            # checks on 'values'
            assert all(isinstance(v, list) for v in values), "each entry in values must be a list."
            
            # checks on 'random_prob'
            assert all(isinstance(v, list) for v in prob_mod), "each entry in random_prob must be a list."

            # transform prob_mod if needed
            random_prob = {}
            for i in range(len(prob_mod)):
                random_prob[i] = []
                if prob_mod[i][0] == 'uniform':
                    random_prob[i] = [1.0 / (len(values[i]))] * (len(values[i]))
                else:
                    random_prob[i] = prob_mod[i]

            # calculate concept values and transform images accordingly
            idx_color = concepts_used.index('colors')
            values[idx_color] = [color_mapping[c] for c in values[idx_color]]


            selected_concepts = {}
            for i in range(len(concepts_used)):
                selected_concepts[concepts_used[i]] = assign_random_values(selected_targets, 
                                                                           random_prob = random_prob[i], 
                                                                           values = values[i])
                
            idx_scale = concepts_used.index('scales') if 'scales' in concepts_used else None
            idx_degree = concepts_used.index('degrees') if 'degrees' in concepts_used else None
            colored_data = transform_images(selected_data, 
                                            transformations=["colorize", "affine"], 
                                            colors= selected_concepts[concepts_used[idx_color]],
                                            degrees= selected_concepts[concepts_used[idx_degree]] if idx_degree is not None else None, 
                                            scales= selected_concepts[concepts_used[idx_scale]] if idx_scale is not None else None)

        else:
            raise ValueError(f"Unknown coloring mode: {m}")

        # assign to the main tensors and dict
        embeddings[start_idx:end_idx] = colored_data
        for k, v in selected_concepts.items():
            if k not in concepts:
                concepts[k] = torch.zeros(N, dtype=v.dtype)
            concepts[k][start_idx:end_idx] = v 
        coloring_mode[start_idx:end_idx] = [split] * selected_data.shape[0]
        
        start_idx = end_idx

    return embeddings, concepts, targets, coloring_mode
