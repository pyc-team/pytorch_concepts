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
    DEFAULT_MODEL_ID,
    KNOWN_CONCEPTS_URL,
    load_steerling_config,
    get_steerling_tokenizer,
    load_steerling_known_head_weights,
    load_steerling_unknown_head_weights,
    load_steerling_lm_head_weights,
    load_steerling_backbone_weights,
    load_steerling_concepts,
    load_steerling_concept_names,
    load_steerling_concept_map,
)
from .steerling_backbone import SteerlingBackbone
from .steerling_encoder import (
    SteerlingLatentToConcept,
    SteerlingConceptExogenousToLatent,
    SteerlingConceptsToLatentEmbeddings,
    SteerlingLatentToLatentFusion,
)
from .steerling_decoder import (
    SteerlingLMHead,
    prepare_generation_sequence,
    prepare_steerling_evidence,
    print_concepts,
)

__all__ = [
    # Utils / hub
    "DEFAULT_MODEL_ID",
    "KNOWN_CONCEPTS_URL",
    "load_steerling_config",
    "get_steerling_tokenizer",
    "load_steerling_known_head_weights",
    "load_steerling_unknown_head_weights",
    "load_steerling_lm_head_weights",
    "load_steerling_backbone_weights",
    "load_steerling_concepts",
    "load_steerling_concept_names",
    "load_steerling_concept_map",
    # Backbone
    "SteerlingBackbone",
    # Encoder
    "SteerlingLatentToConcept",
    "SteerlingConceptExogenousToLatent",
    "SteerlingConceptsToLatentEmbeddings",
    "SteerlingLatentToLatentFusion",
    # Decoder
    "SteerlingLMHead",
    "prepare_generation_sequence",
    "prepare_steerling_evidence",
    "print_concepts",
]
