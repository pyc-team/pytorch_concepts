"""
SteerlingLowLevelModel — end-to-end concept bottleneck demo
============================================================

Tokenize a text prompt and run it through ``SteerlingLowLevelModel``,
which internally wires backbone → known/unknown concept heads →
concept embedding mixers + residual correction → LM head.

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


def print_steerling_runtime_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unknown_embeddings = (
        None
        if model.unknown_embeddings is None
        else tuple(model.unknown_embeddings.shape)
    )
    lm_head_weight = tuple(model.lm_head.weight.shape)
    tokenizer = model.tokenizer

    rows = [
        ("device", model.device),
        ("dtype", next(model.parameters()).dtype),
        ("total_params", f"{total:,}"),
        ("trainable_params", f"{trainable:,}"),
        ("pretrained_components", model.pretrained_components),
        ("frozen_components", model.freeze_components),
        ("backbone_model_id", model.backbone.model_id),
        ("backbone_out_features", model.backbone.out_features),
        ("known_head_factorized", model.known_concept_head.factorize),
        ("unknown_head_factorized", getattr(model.unknown_concept_head, "factorize", None)),
        ("known_head_attention", model.known_concept_head.use_attention),
        ("unknown_head_attention", getattr(model.unknown_concept_head, "use_attention", None)),
        ("known_embeddings", tuple(model.known_embeddings.shape)),
        ("unknown_embeddings", unknown_embeddings),
        ("lm_head_weight", lm_head_weight),
        ("tokenizer_vocab_size", tokenizer.vocab_size),
        ("mask_token_id", tokenizer.mask_token_id),
        ("bos_token_id", tokenizer.bos_token_id),
        ("eos_token_id", tokenizer.eos_token_id),
        ("pad_token_id", tokenizer.pad_token_id),
    ]

    print("\nSteerling runtime:")
    for key, value in rows:
        print(f"  {key:<24} {value}")


device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Instantiate the low-level model ────────────────────────────
# Construct directly in bf16 to halve the CPU-RAM peak during weight load
# (destination tensors are 16 GB instead of 32 GB; peak ≈ 16 + 16 = 32 GB
# instead of 32 + 16 = 48 GB). Restore default dtype afterwards.
_prev_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    model = SteerlingLowLevelModel(
        use_unknown=True,
        use_epsilon_correction=False
    )
finally:
    torch.set_default_dtype(_prev_default_dtype)
model.to(device=device, dtype=torch.bfloat16)
model.eval()
print(model)
print_steerling_config(model)
print_steerling_runtime_info(model)

prompt = "As an italian living abroad in the US, I particularly miss"
n_new_tokens = 20

# ── 2. Sanity check: single forward on the prompt only ────────────
input_ids, _, _ = model.prepare_input(prompt, n_new_tokens=0)
input_ids = input_ids.to(device)
print(f"\nPrompt: {prompt!r}")
print(f"Tokens: {input_ids.shape}")

with torch.no_grad():
    out = model(input_ids)

print(f"Next-token logits:      {out['out_tokens'].shape}")
print(f"Known concept logits:   {out['known_concepts'].shape}")
print(f"Unknown concept logits: {out['unknown_concepts'].shape}")
print(f"Known mixed (k_hat):    {out['known_mixed'].shape}")
print(f"Unknown mixed (u_hat):  {out['unknown_mixed'].shape}")
print(f"Epsilon correction:     {out['epsilon'].shape}, values: {out['epsilon'][0, -1, :5]}")
print(f"Reconstructed latent:   {out['reconstructed_latent'].shape}")

# Top-5 known concepts at the last token of the prompt
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
df = print_concepts(out["known_concepts"][0, -1:], topk=5)   # last position only
print("\nTop-5 known concepts at last prompt token:")
print(df.to_string(index=False))

# ── 3. Full masked diffusion generation ───────────────────────────
model.generate(prompt, n_new_tokens=n_new_tokens, verbose=True)
