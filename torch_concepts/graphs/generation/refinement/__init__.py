"""Export graph-to-graph edge orientation and cycle removal.

refine_llm returns a callable; remove_weakest_cycles and dfs_remove_cycles
are callables themselves. Pass one or an ordered list/tuple to refinement;
DAG validation follows the complete sequence.
"""

from .refinements import (
    refine_llm,
    remove_weakest_cycles,
    dfs_remove_cycles,
)

__all__ = [
    'refine_llm',
    'remove_weakest_cycles',
    'dfs_remove_cycles',
]
