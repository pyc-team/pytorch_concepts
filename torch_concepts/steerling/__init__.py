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
    get_steerling_tokenizer,
    load_steerling_weights,
    load_steerling_concept_names,
    top_concepts,
)
from .steerling_configs import (
    DEFAULT_MODEL_ID,
    PYTORCH_CONCEPTS_CONCEPT_DEFAULTS,
    PYTORCH_CONCEPTS_MODEL_DEFAULTS,
    load_steerling_hub_config,
    resolve_steerling_configs,
)
from .steerling_backbone import CausalDiffusionTextBackbone
from .model.steerling_low import SteerlingLowLevelModel
from .model.steerling import SteerlingModel
from .steerling_encoder import SteerlingLatentToConcept
from .steerling_predictor import MixFactorizedConceptExogenousToConcept

__all__ = [
    # Utils / hub
    "DEFAULT_MODEL_ID",
    "KNOWN_CONCEPTS_URL",
    "load_steerling_hub_config",
    "get_steerling_tokenizer",
    "load_steerling_weights",
    "load_steerling_concept_names",
    "top_concepts",
    "PYTORCH_CONCEPTS_CONCEPT_DEFAULTS",
    "PYTORCH_CONCEPTS_MODEL_DEFAULTS",
    "resolve_steerling_configs",
    # out-of-the-box model
    "SteerlingModel",
    "SteerlingLowLevelModel",
    # Backbone
    "CausalDiffusionTextBackbone",
    # Encoder
    "SteerlingLatentToConcept",
    # Predictor
    "MixFactorizedConceptExogenousToConcept",
]
