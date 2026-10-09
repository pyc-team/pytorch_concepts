"""DAGMA-CGM trainable weighted adjacency with straight-through thresholding.

Adjacency [i, j] represents i -> j. The source owns weights, optional pair
orientation logits and a registered edge mask. Raw adjacency can contain
cycles; the source provides neither a training loop nor an acyclicity loss.
Use a model-specific objective and, if needed, cycle removal during eval.
"""

from __future__ import annotations

from typing import List, Optional
from numbers import Integral

import torch
from torch import nn

from ...base.learnable import GraphGeneratorLearnable, GraphGeneratorLearnableSpec


# ------------------------------------------------------------------
# DAGMA-CGM: CausalCGM's modified DAGMA adjacency
# ------------------------------------------------------------------
def _dagma_cgm_forward(
    generator: GraphGeneratorLearnable, concept_values, concept_names, concept_descriptions,
) -> torch.Tensor:
    """Compute weighted adjacency with a straight-through threshold gate.

    Take absolute masked fc1 weights. For each edges_to_check pair, add a sigmoid
    orientation logit to its source-to-target entry and a complementary score
    to the reverse entry. These pair additions occur after masking and therefore
    are masked again so task-outgoing and self-edge constraints take precedence.

    The forward gate is hard (weight > threshold); backward uses
    sigmoid(5 * (weight - threshold)). Surviving edges keep their weight rather
    than becoming binary. No acyclicity penalty, refinement or cache is applied.
    concept_values is unused: adjacency depends on learned weights only.
    """
    weights = (generator.fc1.weight * generator.edge_mask).abs()
    # Ambiguous pairs are parameterized once, then completed in the opposite
    # direction so each pair keeps a complementary score.
    for source, target in generator.edges_to_check:
        weights[source, target] += torch.sigmoid(
            generator.edge_matrix[source, target]
        )
    for source, target in generator.edges_to_check:
        weights[target, source] += 1 - weights[source, target]
    weights = weights * generator.edge_mask
    soft = torch.sigmoid(5 * (weights - generator.threshold))
    hard = (weights > generator.threshold).to(weights.dtype)
    return weights * (soft + (hard - soft).detach())


@GraphGeneratorLearnable.register_source(
    "DAGMA_CGM", names=["dagma_cgm"],
)
def _load_dagma_cgm_source(
    generator: GraphGeneratorLearnable,
    name: str,
    n_concepts: int,
    n_tasks: int = 0,
    task_indices: Optional[List[int]] = None,
    threshold: float = 0.02,
    no_out_task: bool = True,
    edges_to_check=None,
) -> GraphGeneratorLearnableSpec:
    """Attach DAGMA-CGM trainable adjacency state and its forward callback.

    Parameters
    ----------
    generator : GraphGeneratorLearnable
        PyTorch module receiving fc1, edge_matrix, edge_mask and method options.
    name : {"dagma_cgm"}
        Registered method under source="DAGMA_CGM".
    n_concepts : int
        Number of graph nodes, including task nodes. Names are supplied at call time.
    n_tasks : int, default 0
        If task_indices is omitted, mark the final n_tasks nodes as tasks.
    task_indices : list[int], optional
        Explicit task indices; determines the stored task count.
    threshold : float, default 0.02
        Cutoff used by the straight-through gate; values equal to it are removed.
    no_out_task : bool, default True
        Mask outgoing fc1 edges from task nodes. Self-edges are always masked.
        The mask also applies after explicit edges_to_check adjustments.
    edges_to_check : sequence of (int, int), optional
        Source/target indices for pairs with explicit orientation logits.
        Supply pairs of valid, distinct node indices.

    Returns
    -------
    GraphGeneratorLearnableSpec
        Differentiable callback using the weights initialized by nn.Linear.

    Notes
    -----
    fc1.weight and edge_matrix are parameters; edge_mask is a registered buffer.
    The enclosing model owns optimization and the objective. This source does
    not supply the full DAGMA optimization procedure or guarantee a DAG.
    """
    if name != "dagma_cgm":
        raise ValueError("The DAGMA_CGM source supports only name='dagma_cgm'.")
    if not isinstance(n_concepts, Integral) or isinstance(n_concepts, bool) or n_concepts < 1:
        raise ValueError("n_concepts must be a positive integer.")
    if not isinstance(n_tasks, Integral) or isinstance(n_tasks, bool) or not 0 <= n_tasks <= n_concepts:
        raise ValueError("n_tasks must be between zero and the number of nodes.")
    indices = list(range(n_concepts - n_tasks, n_concepts)) if task_indices is None else list(task_indices)
    if len(set(indices)) != len(indices) or any(
        not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < n_concepts for i in indices
    ):
        raise ValueError("task_indices must contain unique valid node indices.")
    generator.n_concepts = n_concepts
    generator.task_indices = indices
    generator.n_tasks = len(indices)
    generator.fc1 = nn.Linear(
        generator.n_concepts, generator.n_concepts, bias=False
    )
    generator.edge_matrix = nn.Parameter(torch.zeros(
        generator.n_concepts, generator.n_concepts
    ))
    generator.no_out_task = bool(no_out_task)
    generator.edges_to_check = list(edges_to_check or [])
    pair_error = "edges_to_check must contain pairs of valid, distinct node indices."
    for pair in generator.edges_to_check:
        try:
            source, target = pair
        except (TypeError, ValueError):
            raise ValueError(pair_error) from None
        if any(not isinstance(index, Integral) or isinstance(index, bool)
               or not 0 <= index < generator.n_concepts for index in (source, target)) or source == target:
            raise ValueError(pair_error)
    edge_mask = torch.ones(generator.n_concepts, generator.n_concepts)
    edge_mask.fill_diagonal_(0)
    if no_out_task and generator.task_indices:
        edge_mask[generator.task_indices, :] = 0
    generator.register_buffer("edge_mask", edge_mask)
    generator.threshold = float(threshold)
    return GraphGeneratorLearnableSpec(
        forward=_dagma_cgm_forward,
        weights=generator.fc1.weight,
    )
