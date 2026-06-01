"""
SteerlingModel — PGM concept bottleneck demo
============================================

Tokenize a text prompt and run it through ``SteerlingModel``,
which wires backbone → known/unknown concept heads → concept embedding
mixers + residual correction → LM head through a ``ProbabilisticModel`` +
``DeterministicInference``.

Requirements:
    pip install steerling huggingface_hub safetensors

Note:
    First run downloads ~16 GB of model weights (cached by HF Hub).
"""

import torch
from torch_concepts.steerling import SteerlingModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the model ──────────────────────────────────────
model = SteerlingModel(
    pretrained_components=['backbone', 'known_head', 'unknown_head', 'lm_head'],
    freeze_components=['backbone', 'known_head', 'unknown_head', 'lm_head'],
    use_unknown=True,
    use_epsilon_correction=False,
)
model.to(device=device)
model.eval()
print(model)
model.print_config()

prompt = "As an italian living abroad in the US, I particularly miss"
n_new_tokens = 1

# ── 2. Sanity check: single forward on the prompt only ────────────
input_ids, _, _ = model.build_input(prompt, n_new_tokens=0)
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
print(f"Epsilon correction:     {parts['epsilon'].shape}, values: {parts['epsilon'][0, -1, :5]}")
print(f"Reconstructed latent:   {parts['h_bar'].shape}")
print(f"Next-token scores:      {parts['new_token'].shape}")

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, verbose=True)
