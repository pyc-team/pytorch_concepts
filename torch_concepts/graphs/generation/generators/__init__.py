"""Register the built-in static and learnable graph implementations.

Importing the subpackages runs their register_source decorators. Public
construction uses GraphGeneratorStatic and GraphGeneratorLearnable from
``torch_concepts.graphs``; implementation callbacks remain internal.
"""

from . import static
from . import learnable

__all__ = [
]
