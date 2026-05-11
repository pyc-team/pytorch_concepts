"""
SteerlingLowLevelModel — end-to-end concept bottleneck demo
============================================================

Tokenize a text prompt and run it through ``SteerlingLowLevelModel``,
which internally wires backbone → known/unknown concept heads →
concept-to-latent decoder → LM head.

Requirements:
    pip install steerling huggingface_hub safetensors

Note:
    First run downloads ~16 GB of model weights (cached by HF Hub).
"""

import logging

# Reduce HTTP request noise from httpx (used by huggingface_hub)
logging.getLogger("httpx").setLevel(logging.WARNING)

import torch
import pandas as pd
from torch_concepts.steerling import SteerlingLowLevelModel, print_concepts

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the high-level model ───────────────────────────
model = SteerlingLowLevelModel(use_unknown=True, compact=False)
model.to(device)
model.eval()
print(model)

prompt = "As an italian living abroad in the US, I particularly miss"
n_new_tokens = 20

# ── 2. Sanity check: single forward on the prompt only ────────────
input_ids, _, _ = model.prepare_input(prompt, n_new_tokens=0)
input_ids = input_ids.to(device)
print(f"\nPrompt: {prompt!r}")
print(f"Tokens: {input_ids.shape}")

with torch.no_grad():
    out = model(input_ids, unknown_path=True)

print(f"Next-token logits:      {out['out_tokens'].shape}")
print(f"Known concept logits:   {out['known_concepts'].shape}")
print(f"Unknown concept logits: {out['unknown_concepts'].shape}")
print(f"Reconstructed latent:   {out['reconstructed_latent'].shape}")

# Top-5 known concepts at the last token of the prompt
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
df = print_concepts(out["known_concepts"][0, -1:], topk=5)   # last position only
print("\nTop-5 known concepts at last prompt token:")
print(df.to_string(index=False))

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, unknown_path=True, verbose=True)
