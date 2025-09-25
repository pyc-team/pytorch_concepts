from collections import Counter
from typing import Dict, Union, List
import torch, math
import logging

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
                logging.warning(
                    "Added {} to the diagonal of the covariance matrix.".format(
                        num_added
                    )
                )
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
