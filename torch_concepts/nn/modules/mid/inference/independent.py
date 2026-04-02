"""Independent training inference."""

from typing import List, Dict

import torch

from .deterministic import DeterministicInference


class IndependentInference(DeterministicInference):
    """
    Independent training inference.

    This is a convenience subclass of :class:`DeterministicInference` that
    forces ``p=1``, so ground truth concepts are always propagated to
    downstream predictors during training.

    Equivalent to ``DeterministicInference(..., p=1.0)``.

    Parameters
    ----------
    probabilistic_model : ProbabilisticModel
        The probabilistic model to perform inference on.
    graph_learner : BaseGraphLearner, optional
        Optional graph structure learner.
    detach : bool, default False
        If True, detach concept predictions (without GT) before propagation.
    lazy : bool, default False
        If True, only compute ancestors of the queried concepts.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import IndependentInference
        >>>
        >>> # Create inference engine
        >>> inference = IndependentInference(pgm)
        >>>
        >>> # Ground truth in index format: one column per concept
        >>> gt_tensor = torch.tensor([[1, 0]])  # [A_idx, B_idx]
        >>> out = inference.query(['A', 'B'], evidence={'input': x}, 
        ...                       ground_truth=gt_tensor, concept_names=['A', 'B'])
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['p'] = 1.0
        super().__init__(*args, **kwargs)
