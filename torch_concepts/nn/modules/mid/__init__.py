"""
Mid-level API for torch_concepts.

.. warning::
    This module contains **EXPERIMENTAL** mid-level APIs that are subject to change.
    The interfaces and functionality may be modified or removed in future versions
    without a deprecation period. Use at your own risk in production code.

"""
import warnings

# Issue a warning when this module is imported
warnings.warn(
    "The 'torch_concepts.nn.mid' module contains experimental APIs that are unstable "
    "and subject to change without notice. If you are using these classes intentionally, "
    "be aware that breaking changes may occur in future releases. "
    "Consider using the high-level API (torch_concepts.nn.high) for stable interfaces.",
    FutureWarning,
    stacklevel=2
)

__all__: list[str] = []
