"""Public graph generation, initialization and refinement API.

Import generators and utilities from ``torch_concepts.graphs``. Both static and
learnable generators return ConceptGraphs. Learnable training skips refinement;
eval returns a cached, refined and validated final graph.
Learnable generators are PyTorch modules supporting autograd. Dataset
precomputation retains and caches graphs only for static generators.
Adjacency entry [i, j] denotes the edge from node i to node j.
"""

from .generation import (
    GraphGenerator,
    GraphGeneratorSpec,
    GraphGeneratorStatic,
    GraphGeneratorStaticSpec,
    GraphGeneratorLearnable,
    GraphGeneratorLearnableSpec,
    initialize_from_entropy,
    refine_llm,
    remove_weakest_cycles,
    dfs_remove_cycles,
)

__all__ = [
    'GraphGenerator',
    'GraphGeneratorSpec',
    'GraphGeneratorStatic',
    'GraphGeneratorStaticSpec',
    'GraphGeneratorLearnable',
    'GraphGeneratorLearnableSpec',
    'initialize_from_entropy',
    'refine_llm',
    'remove_weakest_cycles',
    'dfs_remove_cycles',
]
