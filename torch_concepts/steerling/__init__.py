"""
Steerling integration for PyTorch Concepts.

This package provides utilities for working with the Steerling family
of interpretable language models, including backbone, encoder, decoder,
and hub/config helpers.
"""

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
)
from .steerling_backbone import SteerlingBackbone
from .steerling_encoder import SteerlingLatentToConcept, SteerlingConceptExogenousToLatent
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
    # Backbone
    "SteerlingBackbone",
    # Encoder
    "SteerlingLatentToConcept",
    "SteerlingConceptExogenousToLatent",
    # Decoder
    "SteerlingLMHead",
    "prepare_generation_sequence",
    "prepare_steerling_evidence",
    "print_concepts",
]
