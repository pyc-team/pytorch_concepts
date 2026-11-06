"""
CBIS-DDSM: a simple concept bottleneck model
===========================================

Use breast density and lesion subtlety as concepts to predict the dataset's
three pathology classes: benign, benign without callback, and malignant.
A small CNN and Lightning handle training and testing. This example uses
256 observations from the official splits; data downloads on first use.
"""

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Trainer
from torchmetrics.classification import MulticlassAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import CBISDDSMDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, ConceptMetrics


def preprocess(image):
    """Normalize a native grayscale image and resize it for batching."""
    image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
    return F.interpolate(
        image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False,
    ).squeeze(0)


def main():
    seed_everything(42)

    # Load a few native categorical concepts; no label conversion is needed.
    datamodule = CBISDDSMDataModule(
        root=str(DATA_ROOT / "cbis_ddsm"),
        concept_subset=["breast_density", "subtlety", "pathology"],
        transform=preprocess, max_samples=256, batch_size=32, seed=42,
    )

    # Image -> density and subtlety -> pathology.
    backbone = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    model = ConceptBottleneckModel(
        input_size=datamodule.n_features, annotations=datamodule.annotations,
        task_names=["pathology"], backbone=backbone, latent_size=32,
        lightning=True,
        loss=ConceptLoss(categorical=nn.CrossEntropyLoss()),
        metrics=ConceptMetrics(
            annotations=datamodule.annotations,
            categorical={"accuracy": MulticlassAccuracy},
            summary=False, per_concept=["pathology"],
        ),
        optim_class=torch.optim.Adam, optim_kwargs={"lr": 1e-3},
    )

    trainer = Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
