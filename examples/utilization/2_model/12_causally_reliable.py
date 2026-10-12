"""
Causally reliable concept bottleneck on Asia.

Use native dataset concepts and a GES + LLM graph, then train and evaluate.
"""

# %%
from functools import partial
from unittest.mock import patch

from pathlib import Path

import matplotlib.style as mpl_style

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style


from torch_concepts.graphs import GraphGeneratorStatic, dfs_remove_cycles, refine_llm
from torch_concepts.llm_backends import LiteLLMBackend
import torch
from pytorch_lightning import Trainer
import torchmetrics

from torch_concepts import seed_everything
from torch_concepts.env import GROQ_API_KEY

from torch_concepts.nn import CausallyReliableConceptBottleneckModel, MLP
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference

seed_everything(42)

ASIA_LABEL_DESCRIPTIONS = {
    "asia": "Recent travel to Asia; an exposure risk factor that can increase the chance of tuberculosis.",
    "smoke": "Smoking status; a risk factor that can increase the chance of lung cancer and bronchitis.",
    "lung": "Lung cancer diagnosis; a disease that can make either true and can contribute to dyspnea.",
    "tub": "Tuberculosis diagnosis; a disease that can make either true and can contribute to dyspnea.",
    "bronc": "Bronchitis diagnosis; a respiratory disease influenced by smoking and able to contribute to dyspnea.",
    "either": "A deterministic logical indicator equal to tub OR lung; it is caused by tuberculosis or lung cancer and can explain an abnormal chest X-ray.",
    "xray": "Abnormal chest X-ray result; an observation influenced by either tuberculosis or lung cancer being present.",
    "dysp": "Dyspnea, or shortness of breath; a symptom influenced by bronchitis and by either tuberculosis or lung cancer.",
}


repo_root = Path(__file__).resolve().parents[3]

PLOTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "causally_reliable" / "plots"
LLM_MODEL = "groq/openai/gpt-oss-20b"
api_key = GROQ_API_KEY
if not api_key:
    raise RuntimeError("Set GROQ_API_KEY in the environment or torch_concepts/env.py before running this example.")

# %%
# Data
# ----

n_samples = 2000
batch_size = 64
# Read CUDA-created caches on CPU; restore torch.load after construction.
with patch("torch.load", partial(torch.load, map_location="cpu")):
    dm = BnLearnDataModule(
        seed=42,
        name="asia",
        root="data/asia_causally_reliable",
        n_gen=n_samples,
        batch_size=batch_size,
        label_descriptions=ASIA_LABEL_DESCRIPTIONS,
    )
dm.setup("fit")
print(f"Dataset size: {dm.n_samples} samples")
print(f"Input shape: {dm.n_features}")
print(f"Concepts: {dm.concept_names}")
print(f"Cardinalities: {dm.annotations.cardinalities}")

batch = next(iter(dm.train_dataloader()))
print(f"inputs:   {tuple(batch['inputs']['x'].shape)}")
print(f"concepts: {tuple(batch['concepts']['c'].shape)}")

# %%
# Graph
# -----
backend = LiteLLMBackend(
    model=LLM_MODEL,
    api_key=api_key,
    temperature=0,
    max_tokens=200,
    retry_on_rate_limit=True,
    max_rate_limit_wait=120.0,
)

# precompute graph with GES + LLM
# setup('fit') is required before precompute_graph to use only training data.
dm.precompute_graph(GraphGeneratorStatic(
    name="ges",
    source="Causallearn",
    refinement=[
        refine_llm(
            llm_backend=backend,
            domain="medical diagnosis",
            repeats=1,
        ),
        dfs_remove_cycles,
    ],
))
graph = dm.dataset.graph
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
graph.plot(PLOTS_DIR / "ges_llm", title="GES + LLM")

annotations = dm.annotations
concept_names = [name for name in dm.concept_names if name != "dysp"]
task_names = ["dysp"]

n_features = dm.n_features[-1]
n_concepts = len(concept_names)
n_tasks = len(task_names)

print(f"Input features: {n_features}")
print(f"Concepts: {n_concepts} - {concept_names}")
print(f"Tasks: {n_tasks} - {task_names}")
print(f"Training samples: {dm.n_samples}")

# %%
# Model
# -----

# Define loss function
loss_fn = ConceptLoss(
    binary = torch.nn.BCEWithLogitsLoss()
)

metrics = ConceptMetrics(
    annotations=annotations,
    summary=True,
    per_concept=True,
    binary = {'accuracy': torchmetrics.classification.BinaryAccuracy()}
)

# Initialize the model
model = CausallyReliableConceptBottleneckModel(
    input_size=n_features,
    annotations=annotations,
    graph=graph,
    embedding_size=8,
    hypernet_hidden_size=8,
    backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
    latent_size=16,
    lightning=True,
    train_inference=DeterministicInference,
    loss=loss_fn,
    metrics=metrics,
    optim_class=torch.optim.AdamW,
    optim_kwargs={'lr': 0.01}
)

print(f"Model created successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Encoder output features: {model.latent_size}")

# %%
# Training
# --------

trainer = Trainer(max_epochs=20)

model.train()
trainer.fit(model, datamodule=dm)
print("Training completed successfully!")

# %%
# Evaluation
# ----------
trainer.test(model=model, datamodule=dm)
