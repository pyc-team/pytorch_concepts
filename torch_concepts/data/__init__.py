"""
Data module for concept-based datasets.

This module provides dataset classes and utilities for working with concept-annotated
data, including various benchmark datasets (MNIST, CelebA, CUB, etc.) and custom
concept datasets.
"""

# Submodules
from . import base
from . import datasets
from . import datamodules
from . import preprocessing
from . import scalers
from . import splitters

# Utilities
from . import utils

# Backbone utilities
from . import backbone

# IO utilities
from . import io

__all__ = [
    # Submodules
    "base",
    "datasets",
    "datamodules",
    "preprocessing",
    "scalers",
    "splitters",

    "utils",
    "backbone",
    "io",
]
