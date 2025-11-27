"""
Custom probability distributions for concept-based models.

This module provides specialized probability distribution classes that extend
PyTorch's distribution framework for use in concept-based neural networks.
"""

from .delta import Delta

__all__ = ["Delta"]