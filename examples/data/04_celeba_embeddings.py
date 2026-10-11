"""
CelebA with Precomputed Embeddings
==================================

CelebA holds face images annotated with 40 binary attributes, which serve as
concepts. A frozen pretrained backbone needs to see each image only once:
``precompute_embeddings`` runs it over the dataset before training, caches the
embeddings next to the data, and swaps them in as the dataset's inputs. Models
then train on the embeddings in seconds, and later runs load them from the
cache.

Data: 1,000 CelebA images (the dataset, ~1.4 GB, is downloaded on first run).
To train through the backbone instead, see ``high_level/10_pretrained_backbone.py``.
"""

# %%
import time

import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import MLP, ConceptBottleneckModel, ConceptLoss, ConceptMetrics

seed_everything(42)

# %%
# Data
# ----
datamodule = CelebADataModule(
    root=str(DATA_ROOT / "celeba"),
    max_samples=1000,
    batch_size=128,
    seed=42,
    splitter=None,  # required with max_samples
)
print(f"images {tuple(datamodule.n_features)}, {datamodule.n_concepts} concepts")

# %%
# Embeddings
# ----------
# Computed and cached on the first run, loaded from the cache afterwards.
backbone = ImageBackbone("resnet18")
start = time.perf_counter()
datamodule.precompute_embeddings(backbone)
seconds = time.perf_counter() - start
print(f"embeddings {tuple(datamodule.n_features)} in {seconds:.1f}s")

# %%
# A model on the embeddings
# -------------------------
model = ConceptBottleneckModel(
    input_size=backbone.out_features,
    annotations=datamodule.annotations,
    task_names=["Attractive"],
    backbone=MLP(backbone.out_features, 128),
    latent_size=128,
    lightning=True,
    loss=ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
    metrics=ConceptMetrics(
        annotations=datamodule.annotations,
        binary={"accuracy": BinaryAccuracy},
        summary=True,
        per_concept=["Attractive"],
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 1e-3},
)
trainer = Trainer(max_epochs=50, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
