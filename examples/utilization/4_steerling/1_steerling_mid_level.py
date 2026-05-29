"""
SteerlingMidLevelModel — PGM concept bottleneck demo
=====================================================

Tokenize a text prompt and run it through ``SteerlingMidLevelModel``,
which wires backbone → known/unknown concept heads → concept embedding
mixers + residual correction → LM head through a ``ProbabilisticModel`` +
``DeterministicInference`` graph, enabling fine-grained concept queries
and future interventions.

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
from torch_concepts.steerling import SteerlingMidLevelModel, top_concepts


def print_steerling_config(model):
    model_cfg = model.model_cfg
    concept_cfg = model.concept_cfg

    rows = [
        ("config_source", model.config_source),
        ("model_type", model_cfg.get("model_type")),
        ("n_layers", model_cfg.get("n_layers")),
        ("n_head", model_cfg.get("n_head")),
        ("n_kv_heads", model_cfg.get("n_kv_heads")),
        ("n_embd", model_cfg.get("n_embd")),
        ("block_size", model_cfg.get("block_size")),
        ("diff_block_size", model_cfg.get("diff_block_size")),
        ("vocab_size", model.vocab_size),
        ("weight_sharing", model_cfg.get("weight_sharing")),
        ("mlp_type", model_cfg.get("mlp_type")),
        ("activation", model_cfg.get("activation")),
        ("use_rms_norm", model_cfg.get("use_rms_norm")),
        ("use_qk_norm", model_cfg.get("use_qk_norm")),
        ("use_rope", model_cfg.get("use_rope")),
        ("rope_base", model_cfg.get("rope_base")),
        ("n_known_concepts", concept_cfg.get("n_concepts")),
        ("n_unknown_concepts", concept_cfg.get("n_unknown_concepts")),
        ("concept_dim", concept_cfg.get("concept_dim")),
        ("use_unknown", concept_cfg.get("use_unknown")),
        ("factorize_unknown", concept_cfg.get("factorize_unknown")),
        ("factorize_rank", concept_cfg.get("factorize_rank")),
        ("use_attention_known", concept_cfg.get("use_attention_known")),
        ("use_attention_unknown", concept_cfg.get("use_attention_unknown")),
        ("use_epsilon_correction", concept_cfg.get("use_epsilon_correction")),
    ]

    print("\nSteerling configuration:")
    for key, value in rows:
        print(f"  {key:<24} {value}")


device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the mid-level model ────────────────────────────
_prev_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    model = SteerlingMidLevelModel(
        use_unknown=True,
        use_epsilon_correction=False
    )
finally:
    torch.set_default_dtype(_prev_default_dtype)
model.to(device=device, dtype=torch.bfloat16)
model.eval()
print(model)
print_steerling_config(model)

prompt = "As an italian living abroad in the US, I particularly miss"
n_new_tokens = 20

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

# Top-5 known concepts at the last token of the prompt
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
df = top_concepts(parts["known_concepts"][0, -1:], topk=5)   # last position only
print("\nTop-5 known concepts at last prompt token:")
print(df.to_string(index=False))

# Targeted PGM query: ask only for the token distribution.
with torch.no_grad():
    token_scores = model(input_ids, query=["new_token"]).probs
print(f"\nTargeted new_token query: {token_scores.shape}")

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, verbose=True)
