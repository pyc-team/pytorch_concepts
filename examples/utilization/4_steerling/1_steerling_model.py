"""
SteerlingModel — PGM concept bottleneck demo
============================================

Tokenize a text prompt and run it through ``SteerlingModel``,
which wires backbone → known/unknown concept heads → concept embedding
mixers + residual correction → LM head through a ``BayesianNetwork`` +
``DeterministicInference``.

Requirements:
    pip install steerling huggingface_hub safetensors

Note:
    First run downloads ~16 GB of model weights (cached by HF Hub).
"""

import torch
import pandas as pd
from torch_concepts.steerling import SteerlingModel, top_concepts

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the model ──────────────────────────────────────
model = SteerlingModel(use_epsilon_correction=True)

model.to(device=device)
model.eval()
print(model)

prompt = "As an italian living abroad in the US, I particularly miss"
n_new_tokens = 20

# ── 2. Sanity check: single forward on the prompt only ────────────
with torch.no_grad():
    input_ids, _, _ = model.build_input(prompt, n_new_tokens=0)
    input_ids = input_ids.to(device)
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {input_ids.shape}")
    out = model(input_ids=input_ids)

# The default query returns known concepts + next token, read from out.params
# by variable name. (The latents h/k_hat/epsilon/h_bar are computed internally
# but not returned unless queried — see below.)
concept_logits = out.params["concepts"]["logits"]      # (1, T, n_known)
token_logits   = out.params["new_token"]["logits"]     # (1, T, vocab)
print(f"Known concept logits:  {tuple(concept_logits.shape)}")
print(f"Next-token logits:     {tuple(token_logits.shape)}")

# Top-5 known concepts at the last prompt token
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
print("\nTop-5 known concepts at last prompt token:")
print(top_concepts(concept_logits[0, -1:], topk=5).to_string(index=False))

# Latents are not in the default output — query them explicitly by name.
with torch.no_grad():
    h_bar = model(input_ids=input_ids, query=["h_bar"]).params["h_bar"]["value"]
print(f"\nReconstructed latent h_bar (queried explicitly): {tuple(h_bar.shape)}")

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, verbose=True)
