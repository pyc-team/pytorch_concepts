from ._version import __version__
from importlib import import_module
from typing import Any

def __getattr__(name: str) -> Any:
    if name in {"data", "nn"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    '__version__'
]
