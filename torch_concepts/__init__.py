from ._version import __version__
from importlib import import_module
from typing import Any

from .concepts.annotations import Annotations, AxisAnnotation
from .concepts.tensor import AnnotatedTensor, ConceptGraph
from .concepts.variable import Variable
from . import nn, distributions
from . import data

def __getattr__(name: str) -> Any:
    if name in {"data", "nn"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",

    "Annotations",
    "AxisAnnotation",
    "AnnotatedTensor",
    "ConceptGraph",
    "Variable",

    "nn",
    "data",
    "distributions",
]
