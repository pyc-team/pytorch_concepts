"""
Functional utilities for concept-based neural networks.

This module provides functional operations for concept manipulation, intervention,
exogenous mixture, and evaluation metrics for concept-based models.
"""
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from typing import Callable, List, Optional, Union, Dict
from torch.nn import Linear
import warnings
import numbers
import torch
import numpy as np
import scipy
from scipy.optimize import Bounds, NonlinearConstraint
from scipy.optimize import minimize as minimize_scipy
from scipy.sparse.linalg import LinearOperator

_constr_keys = {"fun", "lb", "ub", "jac", "hess", "hessp", "keep_feasible"}
_bounds_keys = {"lb", "ub", "keep_feasible"}

from .modules.low.semantic import CMRSemantic


def _default_concept_names(n_concepts: int) -> Dict[int, List[str]]:
    """
    Generate default concept names for a given shape.

    Args:
        shape: List of integers representing the shape of concept dimensions.

    Returns:
        Dict mapping dimension index to list of concept names.
    """
    concept_names = {}
    for dim in range(n_concepts):
        concept_names[dim+1] = [
            f"concept_{dim+1}_{i}" for i in range(n_concepts)
        ]
    return concept_names


def replace_expand_cols(c_emb: torch.Tensor, idx, c_emb_split: torch.Tensor):
    """
    Works for:
      c_emb: [B, D]        with c_emb_split: [B, m, k]
      c_emb: [B, D, E]     with c_emb_split: [B, m, k, E]

    idx:         (m,) indices in D to replace (any order)
    c_emb_split: replacement blocks in SAME order as idx
    returns:     [B, D - m + m*k] or [B, D - m + m*k, E]
    """
    if c_emb.dim() not in (2, 3):
        raise ValueError(f"c_emb must be 2D or 3D, got shape {tuple(c_emb.shape)}")

    B, D = c_emb.shape[:2]
    tail_shape = c_emb.shape[2:]              # () for 2D, (E,) for 3D

    idx = torch.as_tensor(idx, device=c_emb.device, dtype=torch.long)
    idx_sorted, perm = idx.sort()
    m = idx.numel()

    # infer k from c_emb_split
    if c_emb.dim() == 2:
        # c_emb_split: [B, m, k]
        k = c_emb_split.size(2)
        c_emb_split_flat = c_emb_split[:, perm, :].reshape(B, m * k)              # [B, m*k]
    else:
        # c_emb_split: [B, m, k, E]
        k = c_emb_split.size(2)
        c_emb_split_flat = c_emb_split[:, perm, :, :].reshape(B, m * k, *tail_shape)  # [B, m*k, E]

    # counts per original column: 1 for kept, k for replaced
    counts = torch.ones(D, device=c_emb.device, dtype=torch.long)
    counts[idx_sorted] = k
    L = int(counts.sum().item())

    # output position -> original D index
    orig = torch.arange(D, device=c_emb.device).repeat_interleave(counts)  # [L]

    # offset within expanded block
    start = torch.cumsum(counts, 0) - counts                               # [D]
    off = torch.arange(L, device=c_emb.device) - start[orig]               # [L]

    # map original column -> replacement block id (0..m-1), or -1 if not replaced
    col2block = torch.full((D,), -1, device=c_emb.device, dtype=torch.long)
    col2block[idx_sorted] = torch.arange(m, device=c_emb.device)

    is_rep = col2block[orig] >= 0
    rep_col = col2block[orig] * k + off                                    # invalid where ~is_rep
    rep_col_safe = torch.where(is_rep, rep_col, rep_col.new_zeros(rep_col.shape))

    if c_emb.dim() == 2:
        # gather originals
        out = c_emb.gather(1, orig.view(1, L).expand(B, L))                 # [B, L]
        # gather replacements (safe)
        rep = c_emb_split_flat.gather(1, rep_col_safe.view(1, L).expand(B, L))
        # overwrite
        out[:, is_rep] = rep[:, is_rep]
        return out
    else:
        # gather originals
        out = c_emb.gather(1, orig.view(1, L, 1).expand(B, L, *tail_shape)) # [B, L, E]
        # gather replacements (safe)
        rep = c_emb_split_flat.gather(1, rep_col_safe.view(1, L, 1).expand(B, L, *tail_shape))
        # overwrite
        out[:, is_rep, :] = rep[:, is_rep, :]
        return out


def grouped_concept_exogenous_mixture(c_emb: torch.Tensor,
                                      c_scores: torch.Tensor,
                                      groups: list[int]) -> torch.Tensor:
    """
    Vectorized version of grouped concept exogenous mixture.

    Extends  to handle grouped concepts where
    some groups may contain multiple related concepts. Adapted from "Concept Embedding Models:
    Beyond the Accuracy-Explainability Trade-Off" (Espinosa Zarlenga et al., 2022).

    Args:
        c_emb: Concept exogenous of shape (B, n_concepts, emb_size).
        c_scores: Concept scores of shape (B, sum(groups)).
        groups: List of group sizes (e.g., [3, 4] for two groups).

    Returns:
        Tensor: Mixed exogenous of shape (B, len(groups), emb_size // 2).

    Raises:
        AssertionError: If group sizes don't sum to n_concepts.
        AssertionError: If exogenous dimension is not even.

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056

    Example:
        >>> import torch
        >>> from torch_concepts.nn.functional import grouped_concept_exogenous_mixture
        >>>
        >>> # 10 concepts in 3 groups: [3, 4, 3]
        >>> # Embedding size = 20 (must be even)
        >>> batch_size = 4
        >>> n_concepts = 10
        >>> emb_size = 20
        >>> groups = [3, 4, 3]
        >>>
        >>> # Generate random latent and scores
        >>> c_emb = torch.randn(batch_size, n_concepts, emb_size)
        >>> c_scores = torch.rand(batch_size, n_concepts)  # Probabilities
        >>>
        >>> # Apply grouped mixture
        >>> mixed = grouped_concept_exogenous_mixture(c_emb, c_scores, groups)
        >>> print(mixed.shape)
        torch.Size([4, 3, 20])
    """
    B, C, D = c_emb.shape
    assert sum(groups) == C, f"group_sizes must sum to n_concepts. Current group_sizes: {groups}, n_concepts: {C}"

    s = c_scores.unsqueeze(-1)                            # [B, C, 1]

    # Build group ids per concept: [0,0,...,0, 1,1,...,1, ...]
    device = c_emb.device
    G = len(groups)
    gs = torch.as_tensor(groups, device=device)
    group_id = torch.repeat_interleave(torch.arange(G, device=device), gs)  # [C]

    # Weight base embedding by concept scores: [B, C, emb_size]
    eff = s * c_emb

    # Sum weighted exogenous within each group (no loops)
    out = torch.zeros(B, G, D, device=device, dtype=eff.dtype)
    index = group_id.view(1, C, 1).expand(B, C, D)                         # [B, C, E]
    out = out.scatter_add(1, index, eff)                                   # [B, G, E]
    return out


def selection_eval(
    selection_weights: torch.Tensor,
    *predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate concept selection by computing weighted predictions.

    Args:
        selection_weights: Weights for selecting between predictions.
        *predictions: Variable number of prediction tensors to combine.

    Returns:
        Tensor: Weighted combination of predictions.
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction tensor must be provided.")

    product = selection_weights
    for pred in predictions:
        assert pred.shape == product.shape, \
            "Prediction shape mismatch the selection weights."
        product = product * pred

    result = product.sum(dim=-1)

    return result


def linear_equation_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Function to evaluate a set of linear equations with concept predictions.
    In this case we have one equation (concept_weights) for each sample in the
    batch.

    Args:
        concept_weights: Parameters representing the weights of multiple linear
            models with shape (batch_size, memory_size, n_concepts, n_classes).
        c_pred: Concept predictions with shape (batch_size, n_concepts).
        bias: Bias term to add to the linear models (batch_size,
            memory_size, n_classes).

    Returns:
        Tensor: Predictions made by the linear models with shape (batch_size,
            n_classes, memory_size).
    """
    assert concept_weights.shape[-2] == c_pred.shape[-1]
    assert bias is None or bias.shape[-1] == concept_weights.shape[-1]
    y_pred = torch.einsum('bmcy,bc->bym', concept_weights, c_pred)
    if bias is not None:
        # the bias is (b,m,y) while y_pred is (bym) so we invert bias dimension
        y_pred += torch.transpose(bias, -1, -2)
    return y_pred


def linear_equation_expl(
    concept_weights: torch.Tensor,
    bias: torch.Tensor = None,
    concept_names: Dict[int, List[str]] = None,
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extract linear equations from decoded equations embeddings as strings.
    Args:
        concept_weights: Equation embeddings with shape (batch_size,
            memory_size, n_concepts, n_tasks).
        bias: Bias term to add to the linear models (batch_size,
            memory_size, n_tasks).
        concept_names: Concept and task names. If the bias is included, the
            concept names should include the bias name.
    Returns:
        List[Dict[str, Dict[str, str]]]: List of predicted equations as strings.
    """
    if len(concept_weights.shape) != 4:
        raise ValueError(
            "The concept weights must have 4 dimensions (batch_size, "
            "memory_size, n_concepts, n_tasks)."
        )
    if (concept_names is not None
            and concept_weights.shape[-2] != len(concept_names[1])):
        raise ValueError(
            "The concept names must have the same length as the number of "
            "concepts."
        )

    if hasattr(concept_weights, 'concept_names'):
        names = concept_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]
    else:
        # Generate default names for concepts (dimension 2) and tasks (dimension 3)
        if concept_names is None:
            c_names = [f"c_{i}" for i in range(concept_weights.shape[2])]
            t_names = [f"t_{i}" for i in range(concept_weights.shape[3])]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    # add the bias to the concept_weights and c_names
    if bias is not None:
        concept_weights = torch.cat(
            (concept_weights, bias.unsqueeze(-2)),
            dim=-2,
        )
        c_names = c_names + ["bias"]

    batch_size = concept_weights.size(0)
    memory_size = concept_weights.size(1)
    n_concepts = concept_weights.size(2)
    n_tasks = concept_weights.size(3)
    explanation_list = []
    for s_idx in range(batch_size):
        equations_str = defaultdict(dict)  # batch, task, memory_size
        for t_idx in range(n_tasks):
            for mem_idx in range(memory_size):
                eq = []
                for c_idx in range(n_concepts):
                    weight = concept_weights[s_idx, mem_idx, c_idx, t_idx]
                    name = c_names[c_idx]
                    if torch.round(weight.abs(), decimals=2) > 0.1:
                        eq.append(f"{weight.item():.1f} * {name}")
                eq = " + ".join(eq)
                eq = eq.replace(" + -", " - ")
                equations_str[t_names[t_idx]][f"Equation {mem_idx}"] = eq

        explanation_list.append(dict(equations_str))
    return explanation_list


def logic_rule_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    memory_idxs: torch.Tensor = None,
    semantic=CMRSemantic()
) -> torch.Tensor:
    """
    Use concept weights to make predictions based on logic rules.

    Args:
        concept_weights: concept weights with shape (batch_size,
            memory_size, n_concepts, n_tasks, n_roles) with n_roles=3.
        c_pred: concept predictions with shape (batch_size, n_concepts).
        memory_idxs: Indices of rules to evaluate with shape (batch_size,
            n_tasks). Default is None (evaluate all).
        semantic: Semantic function to use for rule evaluation.

    Returns:
        torch.Tensor: Rule predictions with shape (batch_size, n_tasks,
            memory_size)
    """

    assert len(concept_weights.shape) == 5, \
        ("Size error, concept weights should be batch_size x memory_size "
         f"x n_concepts x n_tasks x n_roles. Received {concept_weights.shape}")
    memory_size = concept_weights.size(1)
    n_tasks = concept_weights.size(3)

    # to avoid numerical problem
    concept_weights = concept_weights * 0.999

    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    if memory_idxs is None:
        # cast all to (batch_size, memory_size, n_concepts, n_tasks)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(
            -1,
            memory_size,
            -1,
            n_tasks,
        )
    else:  # cast all to (batch_size, memory_size=1, n_concepts, n_tasks)
        # TODO: memory_idxs never used!
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

    # batch_size, mem_size, n_tasks
    y_per_rule = semantic.disj(
        irrelevance,
        semantic.conj((1 - x), neg_polarity),
        semantic.conj(x, pos_polarity)
    )
    assert (y_per_rule < 1.0).all(), "y_per_rule should be in [0, 1]"

    # performing a conj while iterating over concepts of y_per_rule
    y_per_rule = semantic.conj(
        *[y for y in y_per_rule.split(1, dim=2)]
    ).squeeze(dim=2)

    return y_per_rule.permute(0, 2, 1)


def logic_memory_reconstruction(
    concept_weights: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct tasks based on concept reconstructions, ground truth concepts
    and ground truth tasks.

    Args:
        concept_weights: concept reconstructions with shape (batch_size,
            memory_size, n_concepts, n_tasks).
        c_true: concept ground truth with shape (batch_size, n_concepts).
        y_true: task ground truth with shape (batch_size, n_tasks).

    Returns:
        torch.Tensor: Reconstructed tasks with shape (batch_size, n_tasks,
            memory_size).
    """
    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    # batch_size, mem_size, n_tasks, n_concepts
    c_rec_per_classifier = 0.5 * irrelevance + pos_polarity

    reconstruction_mask = torch.where(
        c_true[:, None, :, None] == 1,
        c_rec_per_classifier,
        1 - c_rec_per_classifier,
    )
    c_rec_per_classifier = reconstruction_mask.prod(dim=2).pow(
        y_true[:, None, :]
    )
    return c_rec_per_classifier.permute(0, 2, 1)


def logic_rule_explanations(
    concept_logic_weights: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extracts rules from rule concept weights as strings.

    Args:
        concept_logic_weights: Rule embeddings with shape
            (batch_size, memory_size, n_concepts, n_tasks, 3).
        concept_names: Concept and task names.

    Returns:
        List[Dict[str, Dict[str, str]]]: Rules as strings.
    """
    if len(concept_logic_weights.shape) != 5 or (
        concept_logic_weights.shape[-1] != 3
    ):
        raise ValueError(
            "The concept logic weights must have 5 dimensions "
            "(batch_size, memory_size, n_concepts, n_tasks, 3)."
        )

    if hasattr(concept_logic_weights, 'concept_names'):
        names = concept_logic_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]
    else:
        # Generate default names for concepts (dimension 2) and tasks (dimension 3)
        if concept_names is None:
            c_names = [f"c_{i}" for i in range(concept_logic_weights.shape[2])]
            t_names = [f"t_{i}" for i in range(concept_logic_weights.shape[3])]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    batch_size = concept_logic_weights.size(0)
    memory_size = concept_logic_weights.size(1)
    n_concepts = concept_logic_weights.size(2)
    n_tasks = concept_logic_weights.size(3)
    # memory_size, n_concepts, n_tasks
    concept_roles = torch.argmax(concept_logic_weights, dim=-1)
    rule_list = []
    for sample_id in range(batch_size):
        rules_str = defaultdict(dict)  # task, memory_size
        for task_id in range(n_tasks):
            for mem_id in range(memory_size):
                rule = []
                for concept_id in range(n_concepts):
                    role = concept_roles[sample_id, mem_id, concept_id, task_id].item()
                    if role == 0:
                        rule.append(c_names[concept_id])
                    elif role == 1:
                        rule.append(f"~ {c_names[concept_id]}")
                    else:
                        continue
                rules_str[t_names[task_id]][f"Rule {mem_id}"] = " & ".join(rule)
        rule_list.append(dict(rules_str))
    return rule_list


def selective_calibration(
    c_confidence: torch.Tensor,
    target_coverage: float,
) -> torch.Tensor:
    """
    Selects concepts based on confidence scores and target coverage.

    Args:
        c_confidence: Concept confidence scores.
        target_coverage: Target coverage.

    Returns:
        Tensor: Thresholds to select confident predictions.
    """
    theta = torch.quantile(
        c_confidence, 1 - target_coverage,
        dim=0,
        keepdim=True,
    )
    return theta


def confidence_selection(
    c_confidence: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Selects concepts with confidence above a selected threshold.

    Args:
        c_confidence: Concept confidence scores.
        theta: Threshold to select confident predictions.

    Returns:
        Tensor: mask selecting confident predictions.
    """
    return torch.where(c_confidence > theta, True, False)


def soft_select(values, temperature, dim=1) -> torch.Tensor:
    """
    Soft selection function, a special activation function for a network
    rescaling the output such that, if they are uniformly distributed, then we
    will select only half of them. A higher temperature will select more
    concepts, a lower temperature will select fewer concepts.

    Args:
        values: Output of the network.
        temperature: Temperature for the softmax function [-inf, +inf].
        dim: dimension to apply the softmax function. Default is 1.

    Returns:
        Tensor: Soft selection scores.
    """

    softmax_scores = torch.log_softmax(values, dim=dim)
    soft_scores = torch.sigmoid(softmax_scores - temperature *
                               softmax_scores.mean(dim=dim, keepdim=True))
    return soft_scores

def completeness_score(
    y_true,
    y_pred_blackbox,
    y_pred_whitebox,
    scorer=roc_auc_score,
    average='macro',
):
    """Calculate the completeness score for the given predictions and true labels.

    Measures how well a concept-based (whitebox) model explains the
    predictions of a blackbox model.  A score of 1.0 indicates that
    the whitebox model fully captures the blackbox's performance.

    Main reference: `"On Completeness-aware Concept-Based Explanations in
    Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_

    Args:
        y_true (torch.Tensor): True labels.
        y_pred_blackbox (torch.Tensor): Predictions from the blackbox model.
        y_pred_whitebox (torch.Tensor): Predictions from the whitebox
            (concept-based) model.
        scorer (callable): Scoring function to evaluate predictions.
            Default is ``roc_auc_score``.
        average (str): Type of averaging to use. Default is ``'macro'``.

    Returns:
        float: Completeness score (whitebox_score / blackbox_score).
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_blackbox_np = y_pred_blackbox.cpu().detach().numpy()
    y_pred_whitebox_np = y_pred_whitebox.cpu().detach().numpy()

    # Compute accuracy or other score using scorer
    blackbox_score = scorer(y_true_np, y_pred_blackbox_np, average=average)
    whitebox_score = scorer(y_true_np, y_pred_whitebox_np, average=average)

    return (whitebox_score) / (blackbox_score + 1e-10)


def intervention_score(
    y_predictor: torch.nn.Module,
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
    intervention_groups: List[List[int]],
    activation: Callable = torch.sigmoid,
    scorer: Callable = roc_auc_score,
    average: str = 'macro',
    auc: bool = True,
) -> Union[float, List[float]]:
    """Compute the effect of concept interventions on downstream task predictions.

    Given a set of intervention groups, the intervention score measures the
    effectiveness of each intervention group on the model's task predictions.

    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/abs/2007.04612>`_

    Args:
        y_predictor (torch.nn.Module): Model that predicts downstream task
            labels.
        c_pred (torch.Tensor): Predicted concept values.
        c_true (torch.Tensor): Ground truth concept values.
        y_true (torch.Tensor): Ground truth task labels.
        intervention_groups (List[List[int]]): List of intervention groups,
            where each group is a list of concept indices to intervene on.
        activation (Callable): Activation function to apply to the model's
            predictions. Default is ``torch.sigmoid``.
        scorer (Callable): Scoring function to evaluate predictions. Default
            is ``roc_auc_score``.
        average (str): Type of averaging to use. Default is ``'macro'``.
        auc (bool): Whether to return the average score across all
            intervention groups. Default is ``True``.

    Returns:
        Union[float, List[float]]: The intervention effectiveness for each
            intervention group, or the average score across all groups when
            ``auc=True``.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()

    # Re-compute the model's predictions for each intervention group
    intervention_effectiveness = []
    for group in intervention_groups:
        # Intervene on the concept values
        c_pred_group = c_pred.clone()
        c_pred_group[:, group] = c_true[:, group]

        # Compute the new model's predictions
        y_pred_group = activation(y_predictor(c_pred_group))

        # Compute the new model's task performance
        intervention_effectiveness.append(scorer(
            y_true_np,
            y_pred_group.cpu().detach().numpy(),
            average=average,
        ))

    # Compute the area under the curve of the intervention curve
    if auc:
        intervention_effectiveness = (
            sum(intervention_effectiveness) / len(intervention_groups)
        )
    return intervention_effectiveness


def _concept_group_ids(
    cardinalities: List[int],
    num_cols: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each weight column to its concept index given per-concept cardinalities.

    A scalar (binary/continuous) concept has cardinality 1 and occupies a single
    column; a categorical concept of cardinality ``m`` occupies ``m`` consecutive
    columns. Returns a ``(num_cols,)`` long tensor of concept ids in ``[0, G)``
    where ``G == len(cardinalities)``.
    """
    cards = [int(c) for c in cardinalities]
    if any(c < 1 for c in cards):
        raise ValueError(
            f"cardinalities must be positive integers, got {cardinalities}"
        )
    total = sum(cards)
    if total != num_cols:
        raise ValueError(
            f"cardinalities sum to {total} but weight has {num_cols} columns"
        )
    return torch.repeat_interleave(
        torch.arange(len(cards), device=device),
        torch.as_tensor(cards, device=device),
    )


def number_of_effective_concepts(
    weight: torch.Tensor,
    threshold: float = 0.0,
    cardinalities: Optional[List[int]] = None,
) -> float:
    """Number of Effective Concepts (NEC) of a linear layer.

    NEC measures the average number of concepts each class relies on for its
    prediction, serving as both a **sparsity** and an **information-leakage
    control** metric: a smaller NEC constrains how much unintended information
    the bottleneck can encode in the downstream prediction.

    Formally, for a weight matrix :math:`W_F \\in \\mathbb{R}^{C \\times k}`
    with :math:`C` classes and :math:`k` concepts:

    .. math::
        \\text{NEC}(W_F) = \\frac{1}{C} \\sum_{i=1}^{C} \\sum_{j=1}^{k}
        \\mathbf{1}[(W_F)_{ij} \\neq 0]

    Main reference: `"VLG-CBM: Training Concept Bottleneck Models with
    Vision-Language Guidance" <https://arxiv.org/abs/2408.01432>`_

    Each weight column is treated as one concept, which is correct for scalar
    (binary / continuous) concepts. For categorical concepts that span several
    columns, pass ``cardinalities`` so each concept is counted once: a class is
    deemed to rely on a concept if **any** of that concept's columns is nonzero.

    Args:
        weight (torch.Tensor): Final linear layer weight matrix of shape
            ``(C, k)`` — C classes, k concept columns.
        threshold (float): Absolute weight magnitude below which a weight is
            treated as zero.  Use ``0.0`` (default) for weights that have
            been pruned to exact zero (e.g. via elastic-net / GLM-SAGA).
            Set a small positive value (e.g. ``1e-6``) when using standard
            L1 regularisation without hard pruning.
        cardinalities (list of int, optional): Number of columns each concept
            occupies, in column order, summing to ``k``. Use ``1`` for scalar
            (binary/continuous) concepts and ``m`` for an ``m``-way categorical
            concept. If ``None`` (default), every column is its own concept.
            If you use the annotation system, pass
            ``cardinalities=annotations.cardinalities``.

    Returns:
        float: Average number of concepts each class relies on.

    Example::

        >>> W = torch.tensor([[1.0, 0.0, -0.5],
        ...                   [0.0, 0.3,  0.0]])
        >>> number_of_effective_concepts(W)  # (2 + 1) / 2
        1.5
        >>> # columns 1-2 form one categorical concept: class 0 uses both
        >>> # concepts (2), class 1 uses only the categorical one (1) -> mean 1.5
        >>> number_of_effective_concepts(W, cardinalities=[1, 2])
        1.5
    """
    mask = (weight.abs() > threshold).float()  # (C, k)
    if cardinalities is None:
        return mask.sum(dim=1).mean().item()

    group_ids = _concept_group_ids(cardinalities, mask.size(1), mask.device)
    num_concepts = len(cardinalities)
    grouped = torch.zeros(
        mask.size(0), num_concepts, device=mask.device, dtype=mask.dtype
    ).index_add(1, group_ids, mask)
    return (grouped > 0).float().sum(dim=1).mean().item()


def number_of_contributing_concepts(
    weight: torch.Tensor,
    concept_activations: torch.Tensor,
    coverage: float = 0.95,
    predicted_class_only: bool = False,
    cardinalities: Optional[List[int]] = None,
) -> float:
    """Number of Contributing Concepts (NCC) of a concept layer.

    NCC is a **decision-level** sparsity (and information-leakage control)
    metric that generalises :func:`number_of_effective_concepts` (NEC).
    Whereas NEC counts nonzero *weights* per class, NCC counts how many
    concepts are actually needed to *explain* each decision, by ranking
    concepts by their **contribution** — the magnitude of the concept logit
    times its class weight — and counting how many top contributors are
    required to cover a fraction :math:`\\tau` of the total contribution.

    For a concept logit vector :math:`g(a^{(i)}) \\in \\mathbb{R}^{k}` for
    image :math:`i` and a weight matrix :math:`W_F \\in \\mathbb{R}^{C \\times k}`,
    the absolute contribution of concept :math:`j` to class :math:`r` is

    .. math::
        u^{(i)}_{j,r} = \\left| [g(a^{(i)})]_{j} \\, (W_F)_{r,j} \\right|.

    Letting :math:`u^{(i)}_{(s),r}` denote the :math:`s`-th largest absolute
    contribution for class :math:`r`, NCC at coverage level
    :math:`\\tau \\in [0, 1]` is

    .. math::
        \\text{NCC}_\\tau = \\frac{1}{|D|\\,C} \\sum_{i=1}^{|D|}
        \\sum_{r=1}^{C} \\min \\Big\\{ \\kappa \\in \\{0, \\dots, k\\} :
        \\sum_{s=1}^{\\kappa} u^{(i)}_{(s),r} \\geq \\tau
        \\sum_{j=1}^{k} u^{(i)}_{j,r} \\Big\\}.

    At :math:`\\tau = 1` NCC reduces to NEC. Lower NCC means more concise,
    less leakage-prone explanations.

    Main reference: `"Learning Concept Bottleneck Models from Mechanistic
    Explanations" (M-CBM, ICLR 2026)
    <https://openreview.net/forum?id=gdEWoxhb70>`_

    Each weight column is treated as one concept, which is correct for scalar
    (binary / continuous) concepts. For categorical concepts that span several
    columns, pass ``cardinalities``; the absolute contributions of a concept's
    columns are **summed** into a single per-concept contribution before ranking.

    Args:
        weight (torch.Tensor): Final linear layer weight matrix of shape
            ``(C, k)`` — C classes, k concept columns.
        concept_activations (torch.Tensor): Concept logits / activations of
            shape ``(N, k)`` for N samples (the inputs to the final layer).
        coverage (float): Fraction :math:`\\tau \\in [0, 1]` of the per-class
            absolute contribution that the top concepts must cover.
            Default: ``0.95``.
        predicted_class_only (bool): If True, average only over each sample's
            predicted class (``argmax`` of the logits) instead of over all C
            classes. Default: ``False``.
        cardinalities (list of int, optional): Number of columns each concept
            occupies, in column order, summing to ``k``. Use ``1`` for scalar
            (binary/continuous) concepts and ``m`` for an ``m``-way categorical
            concept. If ``None`` (default), every column is its own concept.
            If you use the annotation system, pass
            ``cardinalities=annotations.cardinalities``.

    Returns:
        float: Average number of concepts needed to explain a fraction
        ``coverage`` of each decision.

    Example::

        >>> W = torch.tensor([[1.0, 1.0, 0.0],
        ...                   [0.0, 0.0, 2.0]])
        >>> a = torch.tensor([[1.0, 1.0, 1.0]])
        >>> number_of_contributing_concepts(W, a, coverage=1.0)  # == NEC
        1.5
    """
    # Absolute contributions u: (N, C, k) = |activation * weight|.
    contrib = (concept_activations.unsqueeze(1) * weight.unsqueeze(0)).abs()
    if cardinalities is not None:
        # Sum each categorical concept's column contributions into one concept.
        group_ids = _concept_group_ids(
            cardinalities, contrib.size(-1), contrib.device
        )
        num_concepts = len(cardinalities)
        contrib = torch.zeros(
            contrib.size(0), contrib.size(1), num_concepts,
            device=contrib.device, dtype=contrib.dtype,
        ).index_add(2, group_ids, contrib)
    # Sort contributions per (sample, class) in descending order.
    sorted_contrib, _ = torch.sort(contrib, dim=-1, descending=True)
    cumsum = sorted_contrib.cumsum(dim=-1)  # (N, C, k)
    # Use the cumulative total (same accumulation order) so that tau=1 is exact.
    total = cumsum[..., -1:]  # (N, C, 1)
    target = coverage * total
    # Smallest kappa whose top-kappa cumulative sum reaches the target.
    kappa = (cumsum < target).sum(dim=-1) + 1  # (N, C)
    # Degenerate decisions (zero total contribution) need no concepts.
    kappa = torch.where(
        total.squeeze(-1) <= 0, torch.zeros_like(kappa), kappa
    )

    if predicted_class_only:
        logits = concept_activations @ weight.t()  # (N, C)
        pred = logits.argmax(dim=1, keepdim=True)  # (N, 1)
        kappa = kappa.gather(1, pred)  # (N, 1)

    return kappa.float().mean().item()


def cace_score(y_pred_c0, y_pred_c1):
    """Compute the Average Causal Effect (ACE) / Causal Concept Effect (CaCE) score.

    Measures the causal effect of a concept on model predictions:
    ``E[Y | do(C=1)] - E[Y | do(C=0)]``.

    Main reference: `"Explaining Classifiers with Causal Concept Effect
    (CaCE)" <https://arxiv.org/abs/1907.07165>`_

    Args:
        y_pred_c0 (torch.Tensor): Predictions when the concept is inactive
            (``do(C=0)``). Shape: ``(batch_size, num_classes)``.
        y_pred_c1 (torch.Tensor): Predictions when the concept is active
            (``do(C=1)``). Shape: ``(batch_size, num_classes)``.

    Returns:
        torch.Tensor: CaCE score for each class. Shape: ``(num_classes,)``.

    Example:
        >>> import torch
        >>> y_c0 = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        >>> y_c1 = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
        >>> cace_score(y_c0, y_c1)
        tensor([ 0.5000, -0.5000])
    """
    if y_pred_c0.shape != y_pred_c1.shape:
        raise RuntimeError(
            "The shapes of y_pred_c0 and y_pred_c1 must be the same but got "
            f"{y_pred_c0.shape} and {y_pred_c1.shape} instead."
        )
    return y_pred_c1.mean(dim=0) - y_pred_c0.mean(dim=0)


def residual_concept_causal_effect(cace_before, cace_after):
    """Compute the residual concept causal effect.

    Quantifies how much of a concept's causal effect remains after
    intervening on another (inner) concept.  A value close to 1
    indicates that the inner intervention had little impact; values
    close to 0 indicate that the original effect was mostly mediated
    through the inner concept.

    Args:
        cace_before (torch.Tensor): CaCE score **before** the
            do-intervention on the inner concept.
        cace_after (torch.Tensor): CaCE score **after** the
            do-intervention on the inner concept.

    Returns:
        torch.Tensor: Element-wise ratio ``cace_after / cace_before``.

    Example::

        >>> before = torch.tensor([0.5, 0.4])
        >>> after  = torch.tensor([0.1, 0.3])
        >>> residual_concept_causal_effect(before, after)
        tensor([0.2000, 0.7500])
    """
    return cace_after / cace_before

def edge_type(graph, i, j):
    if graph[i,j]==1 and graph[j,i]==0:
        return 'i->j'
    elif graph[i,j]==0 and graph[j,i]==1:
        return 'i<-j'
    elif (graph[i,j]==-1 and graph[j,i]==-1) or (graph[i,j]==1 and graph[j,i]==1):
        return 'i-j'
    elif graph[i,j]==0 and graph[j,i]==0:
        return '/'
    else:
        raise ValueError(f'invalid edge type {i}, {j}')

# graph similairty metrics
def custom_hamming_distance(first, second):
    """Compute the graph edit distance between two partially direceted graphs"""
    first = first.loc[[row for row in first.index if '#virtual_' not in row],
                      [col for col in first.columns if '#virtual_' not in col]]
    first = torch.Tensor(first.values)
    second = second.loc[[row for row in second.index if '#virtual_' not in row],
                        [col for col in second.columns if '#virtual_' not in col]]
    second = torch.Tensor(second.values)
    assert (first.diag() == 0).all() and (second.diag() == 0).all()
    assert first.size() == second.size()
    N = first.size(0)
    cost = 0
    count = 0
    for i in range(N):
        for j in range(i, N):
            if i==j: continue
            if edge_type(first, i, j)==edge_type(second, i, j): continue
            else:
                count += 1
                # edge was directed
                if edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i<-j': cost += 1./3.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i->j': cost += 1./3.
                # edge was undirected
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i->j': cost += 1./4. 
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i<-j': cost += 1./4.
                # there was no edge
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i-j': cost += 1./2.
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i->j': cost += 1
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i<-j': cost += 1

                else:  
                    raise ValueError(f'invalid combination of edge types {i}, {j}')
    
    # cost = cost / (N*(N-1))/2
    return cost, count


def prune_linear_layer(linear: Linear, mask: torch.Tensor, dim: int = 0) -> Linear:
    """
    Return a new nn.Linear where inputs (dim=0) or outputs (dim=1)
    have been pruned according to `mask`.

    Args
    ----
    linear : nn.Linear
        Layer to prune.
    mask : 1D Tensor[bool] or 0/1
        Mask over features. True/1 = keep, False/0 = drop.
        - If dim=0: length == in_features
        - If dim=1: length == out_features
    dim : int
        0 -> prune input features (columns of weight)
        1 -> prune output units (rows of weight)
    """
    if not isinstance(linear, Linear):
        raise TypeError("`linear` must be an nn.Linear")

    mask = mask.to(dtype=torch.bool)
    weight = linear.weight
    device = weight.device
    dtype = weight.dtype

    idx = mask.nonzero(as_tuple=False).view(-1)  # indices to KEEP

    if dim == 0:
        if mask.numel() != linear.in_features:
            raise ValueError("mask length must equal in_features when dim=0")

        new_in = idx.numel()
        new_linear = Linear(
            in_features=new_in,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            # keep all rows (outputs), select only kept input columns
            new_linear.weight.copy_(weight[:, idx])
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias)

    elif dim == 1:
        if mask.numel() != linear.out_features:
            raise ValueError("mask length must equal out_features when dim=1")

        new_out = idx.numel()
        new_linear = Linear(
            in_features=linear.in_features,
            out_features=new_out,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            # select only kept output rows
            new_linear.weight.copy_(weight[idx, :])
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias[idx])

    else:
        raise ValueError("dim must be 0 (inputs) or 1 (outputs)")

    return new_linear


def _build_obj(f, x0):
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f_with_jac(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        (grad,) = torch.autograd.grad(fval, x)
        return fval.detach().cpu().numpy(), grad.view(-1).cpu().numpy()

    def f_hess(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            (grad,) = torch.autograd.grad(fval, x, create_graph=True)

        def matvec(p):
            p = to_tensor(p)
            (hvp,) = torch.autograd.grad(grad, x, p, retain_graph=True)
            return hvp.view(-1).cpu().numpy()

        return LinearOperator((numel, numel), matvec=matvec)

    return f_with_jac, f_hess


def _build_constr(constr, x0):
    assert isinstance(constr, dict)
    assert set(constr.keys()).issubset(_constr_keys)
    assert "fun" in constr
    assert "lb" in constr or "ub" in constr
    if "lb" not in constr:
        constr["lb"] = -np.inf
    if "ub" not in constr:
        constr["ub"] = np.inf
    f_ = constr["fun"]
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f(x):
        x = to_tensor(x)
        return f_(x).cpu().numpy()

    def f_jac(x):
        x = to_tensor(x)
        if "jac" in constr:
            grad = constr["jac"](x)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                (grad,) = torch.autograd.grad(f_(x), x)
        return grad.view(-1).cpu().numpy()

    def f_hess(x, v):
        x = to_tensor(x)
        if "hess" in constr:
            hess = constr["hess"](x)
            return v[0] * hess.view(numel, numel).cpu().numpy()
        elif "hessp" in constr:

            def matvec(p):
                p = to_tensor(p)
                hvp = constr["hessp"](x, p)
                return v[0] * hvp.view(-1).cpu().numpy()

            return LinearOperator((numel, numel), matvec=matvec)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                if "jac" in constr:
                    grad = constr["jac"](x)
                else:
                    (grad,) = torch.autograd.grad(f_(x), x, create_graph=True)

            def matvec(p):
                p = to_tensor(p)
                if grad.grad_fn is None:
                    # If grad_fn is None, then grad is constant wrt x, and hess is 0.
                    hvp = torch.zeros_like(grad)
                else:
                    (hvp,) = torch.autograd.grad(grad, x, p, retain_graph=True)
                return v[0] * hvp.view(-1).cpu().numpy()

            return LinearOperator((numel, numel), matvec=matvec)

    return NonlinearConstraint(
        fun=f,
        lb=constr["lb"],
        ub=constr["ub"],
        jac=f_jac,
        hess=f_hess,
        keep_feasible=constr.get("keep_feasible", False),
    )


def _check_bound(val, x0):
    if isinstance(val, numbers.Number):
        return np.full(x0.numel(), val)
    elif isinstance(val, torch.Tensor):
        assert val.numel() == x0.numel()
        return val.detach().cpu().numpy().flatten()
    elif isinstance(val, np.ndarray):
        assert val.size == x0.numel()
        return val.flatten()
    else:
        raise ValueError("Bound value has unrecognized format.")


def _build_bounds(bounds, x0):
    assert isinstance(bounds, dict)
    assert set(bounds.keys()).issubset(_bounds_keys)
    assert "lb" in bounds or "ub" in bounds
    lb = _check_bound(bounds.get("lb", -np.inf), x0)
    ub = _check_bound(bounds.get("ub", np.inf), x0)
    keep_feasible = bounds.get("keep_feasible", False)

    return Bounds(lb, ub, keep_feasible)

#### CODE adapted from https://pytorch-minimize.readthedocs.io/en/latest/_modules/torchmin/minimize_constr.html#minimize_constr

@torch.no_grad()
def minimize_constr(
    f,
    x0,
    constr=None,
    bounds=None,
    max_iter=None,
    tol=None,
    callback=None,
    disp=0,
    **kwargs
):
    """Minimize a scalar function of one or more variables subject to
    bounds and/or constraints.

    .. note::
        This is a wrapper for SciPy's
        `'trust-constr' <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_
        method. It uses autograd behind the scenes to build jacobian & hessian
        callables before invoking scipy. Inputs and objectivs should use
        PyTorch tensors like other routines. CUDA is supported; however,
        data will be transferred back-and-forth between GPU/CPU.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    constr : dict, optional
        Constraint specifications. Should be a dictionary with the
        following fields:

            * fun (callable) - Constraint function
            * lb (Tensor or float, optional) - Constraint lower bounds
            * ub : (Tensor or float, optional) - Constraint upper bounds

        One of either `lb` or `ub` must be provided. When `lb` == `ub` it is
        interpreted as an equality constraint.
    bounds : dict, optional
        Bounds on variables. Should a dictionary with at least one
        of the following fields:

            * lb (Tensor or float) - Lower bounds
            * ub (Tensor or float) - Upper bounds

        Bounds of `-inf`/`inf` are interpreted as no bound. When `lb` == `ub`
        it is interpreted as an equality constraint.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    **kwargs
        Additional keyword arguments passed to SciPy's trust-constr solver.
        See options `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    if max_iter is None:
        max_iter = 1000
    x0 = x0.detach()
    if x0.is_cuda:
        warnings.warn(
            "GPU is not recommended for trust-constr. "
            "Data will be moved back-and-forth from CPU."
        )

    # handle callbacks
    if callback is not None:
        callback_ = callback
        callback = lambda x, state: callback_(
            torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0), state
        )

    # handle bounds
    if bounds is not None:
        bounds = _build_bounds(bounds, x0)

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    # build objective function (and hessian)
    if "jac" in kwargs.keys() and "hess" in kwargs.keys():
        jacobian = kwargs.pop("jac")
        hessian = kwargs.pop("hess")

        def f_with_jac(x):
            x = to_tensor(x)
            fval = f(x)
            grad = jacobian(x)
            return fval.cpu().numpy(), grad.cpu().numpy()

        if type(hessian) == str:
            f_hess = hessian
        else:

            def f_hess(x):
                x = to_tensor(x)

                def matvec(p):
                    p = to_tensor(p)
                    hvp = hessian(x) @ p
                    return hvp.cpu().numpy()

                return LinearOperator((x0.numel(), x0.numel()), matvec=matvec)

    elif "jac" in kwargs.keys():
        _, f_hess = _build_obj(f, x0)
        jacobian = kwargs.pop("jac")

        def f_with_jac(x):
            x = to_tensor(x)
            fval = f(x)
            grad = jacobian(x)
            return fval.cpu().numpy(), grad.cpu().numpy()

    else:
        f_with_jac, f_hess = _build_obj(f, x0)

    # build constraints
    if constr is not None:
        constraints = [_build_constr(constr, x0)]
    else:
        constraints = []

    # optimize
    x0_np = x0.float().cpu().numpy().flatten().copy()
    method = kwargs.pop("method", "trust-constr")  # Default to trust-constr
    if method == "trust-constr":
        result = minimize_scipy(
            f_with_jac,
            x0_np,
            method="trust-constr",
            jac=True,
            hess=f_hess,
            callback=callback,
            tol=tol,
            bounds=bounds,
            constraints=constraints,
            options=dict(verbose=int(disp), maxiter=max_iter, **kwargs),
        )
    elif method == "SLSQP":
        if constr["ub"] == constr["lb"]:
            constr["type"] = "eq"
        elif constr["lb"] == 0:
            constr["type"] = "ineq"
        elif constr["ub"] == 0:
            constr["type"] = "ineq"
            original_fun2 = constr["fun"]
            constr["fun"] = lambda x: -original_fun2(x)
        else:
            raise NotImplementedError(
                "Only equality and inequality constraints around 0 are supported"
            )
        original_fun = constr["fun"]
        original_jac = constr["jac"]
        # scipy's SLSQP backend requires float64 inputs/outputs throughout.
        constr["fun"] = lambda x: original_fun(torch.tensor(x).float()).cpu().numpy().astype("float64")
        constr["jac"] = lambda x: original_jac(torch.tensor(x).float()).cpu().numpy().astype("float64")
        x0_np = x0_np.astype("float64")
        f_slsqp = f_with_jac
        f_with_jac = lambda x: tuple(
            v.astype("float64") for v in f_slsqp(x)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=scipy.optimize._optimize.__name__,
            )
            result = minimize_scipy(
                f_with_jac,
                x0_np,
                method="SLSQP",
                jac=True,
                callback=callback,
                tol=tol,
                bounds=bounds,
                constraints=constr,
                options=dict(maxiter=max_iter),
            )

    # convert the important things to torch tensors
    for key in ["fun", "x"]:
        result[key] = torch.tensor(result[key], dtype=x0.dtype, device=x0.device)
    result["x"] = result["x"].view_as(x0)

    return result
