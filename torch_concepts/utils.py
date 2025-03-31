from collections import Counter
from typing import Dict, Union, List


def validate_and_generate_concept_names(
    concept_names: Dict[int, Union[int, List[str]]]
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
            processed_concept_names[dim] = [
                f"concept_{dim}_{i}" for i in range(value)
            ]
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
        explanations: List[Dict[str, str]],
        n=10
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
