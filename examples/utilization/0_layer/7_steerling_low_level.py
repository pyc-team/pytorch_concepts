"""
Text → Backbone → Concept logits  (Steerling-8B)
=================================================

End-to-end example: tokenize a text prompt, pass it through the
pretrained Steerling transformer backbone, then feed the hidden states
into the pretrained concept head to produce concept logits.

Requirements:
    pip install steerling huggingface_hub safetensors

Note:
    First run downloads ~16 GB of model weights (cached by HF Hub).
"""

import logging

# Reduce HTTP request noise from httpx (used by huggingface_hub)
logging.getLogger("httpx").setLevel(logging.WARNING)

import torch
from torch_concepts.steerling import SteerlingBackbone, SteerlingLatentToConcept, print_concepts

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load pretrained backbone (transformer) ─────────────────────
# Note: Steerling-8B requires ~16 GB. Use device="cpu" if GPU memory
# is insufficient, or "cuda" for GPUs with >= 24 GB VRAM.
backbone = SteerlingBackbone(pretrained=True, freeze=True, device="cpu")
print(backbone)

# ── 2. Load pretrained concept head ───────────────────────────────
concept_head = SteerlingLatentToConcept(pretrained=True, freeze=True)
print(f"Concept head: {concept_head.in_latent}→{concept_head.out_concepts}")

# ── 3. Tokenize a simple prompt ──────────────────────────────────
prompt = "The key to understanding artificial intelligence is"
encoded = backbone.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"]
print(f"\nPrompt: {prompt!r}")
print(f"Tokens: {input_ids.shape}")

# ── 4. Backbone forward: tokens → hidden states ──────────────────
with torch.no_grad():
    hidden = backbone(input_ids)  # (1, T, 4096)
    hidden = hidden.float()       # concept head expects float32
print(f"Hidden states: {hidden.shape}")

# ── 5. Concept head forward: hidden → logits ─────────────────────
with torch.no_grad():
    logits = concept_head(hidden)  # (1, T, n_concepts)
print(f"Concept logits: {logits.shape}")

# ── 6. Decode concept logits to human-readable names ──────────────
import pandas as pd
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)

df = print_concepts(logits[0],topk=5) # (T, n_concepts)
print("\nTop-5 concepts per token:")
print(df.to_string(index=False))