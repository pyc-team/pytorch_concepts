"""Register the DAGMA_CGM learnable source under name="dagma_cgm".

The source attaches trainable state to GraphGeneratorLearnable and supplies
its adjacency callback. Optimization and the training objective belong to
the enclosing model or training loop.
"""

from . import dagma_cgm

__all__ = [
]
