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

from torch_concepts.nn.modules.high.models.steerling.steerling_utils import load_steerling_concept_names

# Reduce HTTP request noise from httpx (used by huggingface_hub)
logging.getLogger("httpx").setLevel(logging.WARNING)

import torch
from steerling import ConceptCatalog
from torch_concepts.nn.modules.high.models.steerling import SteerlingLowLevelModelPrototypes

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# ── 1. Instantiate the low-level model ────────────────────────────
# The model builds itself in its native bfloat16 (see the dtype= arg to
# override), halving the CPU-RAM peak during weight load.
model = SteerlingLowLevelModelPrototypes()

model.to(device=device)
model.eval()
print(model)

prompt = "I suspect the bank rejected my loan request because of my gender"

concept_df = ConceptCatalog.load().to_df()
target_concept = "Gender, Sex, and Patriarchy"
target_concept_id = concept_df.index[concept_df["concept_name"] == target_concept].item()

prototypes = {
    target_concept_id: [
        "The train departs platform 4 at 9:15 and makes stops in three towns before reaching the coast.",

        "The committee reviewed the annual budget and approved funding for the new library wing.",

        "Patriarchal structures have historically shaped gender roles and the distribution of power between "
        "men and women, influencing everything from labor division to political representation and reproductive rights."
    ]
}

# ── 3. Full masked diffusion generation ───────────────────────────
n_new_tokens = 20
model.generate(
    prompt,
    n_new_tokens=n_new_tokens,
    prototypes=prototypes,
    verbose=True
)
