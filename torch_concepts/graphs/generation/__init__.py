"""Graph generator APIs and built-in source registration.

Importing this package registers Causallearn and LLM for
GraphGeneratorStatic, and DAGMA_CGM for GraphGeneratorLearnable. It also
exports the source callback contracts, initializers and refinements.
"""

from .base import (
    GraphGenerator,
    GraphGeneratorSpec,
    GraphGeneratorStatic,
    GraphGeneratorStaticSpec,
    GraphGeneratorLearnable,
    GraphGeneratorLearnableSpec,
)
from .initialization import (
    initialize_from_entropy,
)
from .refinement import (
    refine_llm,
    remove_weakest_cycles,
    dfs_remove_cycles,
)
from . import generators

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
