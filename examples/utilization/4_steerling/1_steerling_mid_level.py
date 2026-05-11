"""
SteerlingMidLevelModel — PGM concept bottleneck demo
=====================================================

Tokenize a text prompt and run it through ``SteerlingMidLevelModel``,
which wires backbone → known/unknown concept heads → concept-to-latent
decoder → LM head through a ``ProbabilisticModel`` + ``DeterministicInference``
graph, enabling fine-grained concept queries and future interventions.

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
from torch_concepts.steerling import SteerlingMidLevelModel, print_concepts

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the mid-level model ────────────────────────────
model = SteerlingMidLevelModel(
    use_unknown=True, 
    compact=False, 
    use_epsilon_correction=False
)
model.to(device=device, dtype=torch.bfloat16)
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
    out = model(input_ids)

parts = model.split_full_forward(out.probs)
print(f"Input hidden states:    {parts['input'].shape}")
print(f"Known concept scores:   {parts['known_concepts'].shape}")
if "unknown_concepts" in parts:
    print(f"Unknown concept scores: {parts['unknown_concepts'].shape}")
print(f"Known latent mix:       {parts['k_hat'].shape}")
if "u_hat" in parts:
    print(f"Unknown latent mix:     {parts['u_hat'].shape}")
print(f"Reconstructed latent:   {parts['h_bar'].shape}")
print(f"Next-token scores:      {parts['new_token'].shape}")

# Top-5 known concepts at the last token of the prompt
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
df = print_concepts(parts["known_concepts"][0, -1:], topk=5)   # last position only
print("\nTop-5 known concepts at last prompt token:")
print(df.to_string(index=False))

# Targeted PGM query: ask only for the token distribution.
with torch.no_grad():
    token_scores = model(input_ids, query=["new_token"]).probs
print(f"\nTargeted new_token query: {token_scores.shape}")

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, verbose=True)
