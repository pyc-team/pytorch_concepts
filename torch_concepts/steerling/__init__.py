"""
Steerling integration for PyTorch Concepts.

This package provides utilities for working with the Steerling family
of interpretable language models, including backbone, encoder, decoder,
and hub/config helpers.
"""

import os

# Default to eager mode for Steerling to avoid Triton/Inductor failures on
# some CUDA driver stacks. Override with:
#   TORCH_CONCEPTS_ENABLE_TORCH_COMPILE=1
if os.environ.get("TORCH_CONCEPTS_ENABLE_TORCH_COMPILE", "0") != "1":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from .steerling_utils import (
    KNOWN_CONCEPTS_URL,
    active_concepts,
    get_steerling_tokenizer,
    load_steerling_known_head_weights,
    load_steerling_unknown_head_weights,
    load_steerling_lm_head_weights,
    load_steerling_backbone_weights,
    load_steerling_concepts,
    load_steerling_concept_names,
    load_steerling_concept_map,
    prepare_generation_sequence,
    prepare_steerling_evidence,
    print_concepts,
)
from .steerling_configs import (
    DEFAULT_MODEL_ID,
    PYTORCH_CONCEPTS_CONCEPT_DEFAULTS,
    PYTORCH_CONCEPTS_MODEL_DEFAULTS,
    load_steerling_hub_config,
    normalize_steerling_components,
    resolve_steerling_configs,
)
from .steerling_backbone import CausalDiffusionTextBackbone
from .model.steerling_low import SteerlingLowLevelModel
from .model.steerling_mid import SteerlingMidLevelModel
from .steerling_encoder import SteerlingLatentToConcept
from .steerling_predictor import MixFactorizedConceptExogenousToConcept

__all__ = [
    # Utils / hub
    "DEFAULT_MODEL_ID",
    "KNOWN_CONCEPTS_URL",
    "active_concepts",
    "load_steerling_hub_config",
    "get_steerling_tokenizer",
    "load_steerling_known_head_weights",
    "load_steerling_unknown_head_weights",
    "load_steerling_lm_head_weights",
    "load_steerling_backbone_weights",
    "load_steerling_concepts",
    "load_steerling_concept_names",
    "load_steerling_concept_map",
    "prepare_generation_sequence",
    "prepare_steerling_evidence",
    "print_concepts",
    "PYTORCH_CONCEPTS_CONCEPT_DEFAULTS",
    "PYTORCH_CONCEPTS_MODEL_DEFAULTS",
    "normalize_steerling_components",
    "resolve_steerling_configs",
    # out-of-the-box model
    "SteerlingLowLevelModel",
    "SteerlingMidLevelModel",
    # Backbone
    "CausalDiffusionTextBackbone",
    # Encoder
    "SteerlingLatentToConcept",
    # Predictor
    "MixFactorizedConceptExogenousToConcept",
]
