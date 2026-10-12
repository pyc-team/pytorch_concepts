"""Shared graph lifecycle and static/learnable source contracts.

GraphGenerator provides common registration, context, refinement and
validation. Instantiate GraphGeneratorStatic or GraphGeneratorLearnable;
the shared base cannot be instantiated directly.
"""

from .base import (
    GraphGenerator,
    GraphGeneratorSpec,
)
from .static import (
    GraphGeneratorStatic,
    GraphGeneratorStaticSpec,
)
from .learnable import (
    GraphGeneratorLearnable,
    GraphGeneratorLearnableSpec,
)

__all__ = [
    'GraphGenerator',
    'GraphGeneratorSpec',
    'GraphGeneratorStatic',
    'GraphGeneratorStaticSpec',
    'GraphGeneratorLearnable',
    'GraphGeneratorLearnableSpec',
]
