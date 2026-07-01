"""
Steerling integration for PyTorch Concepts.

Utilities for the Steerling family of interpretable language models: the
backbone, the concept encoder and mixer layers, the high- and low-level
models, and hub/config helpers.
"""

import os
import warnings

try:
    import conceptarium.env  # noqa: F401 — seeds HF_TOKEN into os.environ
except ImportError:
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        warnings.warn(
            "conceptarium.env not found and HF_TOKEN is not set. "
            "Hub downloads will be unauthenticated.",
            stacklevel=2,
        )

# Default to eager mode for Steerling to avoid Triton/Inductor failures on
# some CUDA driver stacks. Override with:
#   TORCH_CONCEPTS_ENABLE_TORCH_COMPILE=1
if os.environ.get("TORCH_CONCEPTS_ENABLE_TORCH_COMPILE", "0") != "1":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from .steerling_utils import (
    KNOWN_CONCEPTS_URL,
    top_concepts
)
from .steerling_configs import (
    DEFAULT_MODEL_ID,
)
from .model.steerling_low import SteerlingLowLevelModel
from .model.steerling import SteerlingModel

__all__ = [
    # Utils / hub
    "DEFAULT_MODEL_ID",
    "KNOWN_CONCEPTS_URL",
    "top_concepts",
    # out-of-the-box model
    "SteerlingModel",
    "SteerlingLowLevelModel",
]
