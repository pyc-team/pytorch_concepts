"""
torch_concepts: A PyTorch library for concept-based machine learning.

This package provides tools and modules for building concept-based neural networks.
"""
from ._version import __version__
from importlib import import_module
from typing import Any

from .annotations import Annotations, AxisAnnotation
from .nn.modules.utils import GroupConfig
from .nn.modules.mid.constructors.concept_graph import ConceptGraph
from .nn.modules.mid.models.variable import Variable, InputVariable, ExogenousVariable, EndogenousVariable
from .utils import seed_everything
from . import nn, distributions
from . import data

def __getattr__(name: str) -> Any:
    if name in {"data", "nn"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",

    # Data properties
    "Annotations",
    "AxisAnnotation",
    "ConceptGraph",

    # Configuration
    "GroupConfig",

    # Variables
    "Variable",
    "InputVariable",
    "ExogenousVariable",
    "EndogenousVariable",

    "seed_everything",

    "nn",
    "data",
    "distributions",
]
