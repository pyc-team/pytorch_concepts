"""
CBIS-DDSM
=========

CBIS-DDSM holds mammograms with native categorical concepts. This example uses
breast density and lesion subtlety as concepts to predict the dataset's three
pathology classes: benign, benign without callback, and malignant.

A fine-tuned ResNet-18 and Lightning handle training and testing. Data: the
official splits, downloaded on first run.
"""

# %%
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Trainer
from torchmetrics.classification import MulticlassAccuracy

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CBISDDSMDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, ConceptMetrics

seed_everything(42)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(image):
    """Scale a native grayscale image to ImageNet-normalized RGB at 128x128."""
    image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (image.expand(3, -1, -1) - MEAN) / STD


# %%
# Data
# ----
# Native categorical concepts; no label conversion is needed.
dm = CBISDDSMDataModule(
    root=str(DATA_ROOT / "cbis_ddsm"),
    concept_subset=["breast_density", "subtlety", "pathology"],
    transform=preprocess,
    # max_samples=256,
    batch_size=32,
    seed=42,
)
dm.setup()
print(f"Dataset size: {dm.n_samples} samples")
print(f"Image shape: {dm.n_features}")
print(f"Concepts: {dm.concept_names}")
print(f"Cardinalities: {dm.annotations.cardinalities}")

batch = next(iter(dm.train_dataloader()))
print(f"inputs:   {tuple(batch['inputs']['x'].shape)}")
print(f"concepts: {tuple(batch['concepts']['c'].shape)}")

# %%
# A concept bottleneck model
# --------------------------
# Image -> density and subtlety -> pathology.
backbone = ImageBackbone("resnet18", freeze=False)
model = ConceptBottleneckModel(
    input_size=dm.n_features,
    annotations=dm.annotations,
    task_names=["pathology"],
    backbone=backbone,
    lightning=True,
    loss=ConceptLoss(categorical=nn.CrossEntropyLoss()),
    metrics=ConceptMetrics(
        annotations=dm.annotations,
        categorical={"accuracy": (MulticlassAccuracy, {"average": "micro"})},
        summary=False,
        per_concept=["breast_density", "subtlety", "pathology"],
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 1e-4},
)

# %%
# Training
# --------
trainer = Trainer(max_epochs=20, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
