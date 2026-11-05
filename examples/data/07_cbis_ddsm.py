"""
CBIS-DDSM: training a concept bottleneck model
=============================================

Load the official mass training/test splits and train an ImageNet-pretrained
DenseNet121 CBM using native shape, margin, and breast-density labels. The task
is binary malignancy, derived from the dataset's original pathology states.
Missing shape/margin targets are imputed with training-split prevalence as
soft labels. Evaluation scores only originally observed labels.
Lightning trains on every training image; ROC AUC uses every test image.

Data and pretrained weights download on first use. Run with:
``python -u examples/data/07_cbis_ddsm.py``.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

from torch_concepts import AnnotatedTensor, Annotations, seed_everything
from torch_concepts.data import CBISDDSMDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import CompositeLoss, ConceptBottleneckModel, ConceptLoss, ConceptSubset


def resize_crop(image):
    """Scale pixels to [0, 1], square-pad, resize to 224, and replicate RGB."""
    image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
    height, width = image.shape[-2:]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    image = F.pad(image, (left, side - width - left, top, side - height - top))
    image = F.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(0)
    return image.repeat(3, 1, 1)


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # %%
    # Data: preserve official split membership and native concept definitions.
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    datamodule = CBISDDSMDataModule(
        root=str(DATA_ROOT / "cbis_ddsm"), abnormality_type="mass",
        image_type="cropped", val_size=0, batch_size=32, workers=0, seed=42,
        transform=transforms.Compose([resize_crop, normalize]),
        train_transform=transforms.Compose([
            resize_crop, transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10), normalize,
        ]),
    )
    descriptors = datamodule.annotations.resolve(["mass_shape", "mass_margin"])[1]
    binary_names = [name for name in descriptors.labels if not name.endswith(".N/A")]
    concept_names = binary_names + ["breast_density"]
    annotations = datamodule.annotations.subset(concept_names).union_with(
        Annotations(labels=["malignancy"], types=["binary"], states=[["present"]]),
    )
    malignant = datamodule.annotations.concept("pathology").states.index("MALIGNANT")
    pathology = datamodule.native_concepts["pathology"].tensor
    diagnosis = torch.where(torch.isfinite(pathology), (pathology == malignant).float(), torch.nan)
    # Fit imputation on training observations only; preserve the native labels.
    binary_targets = datamodule.native_concepts[binary_names].tensor
    prevalence = binary_targets[datamodule.splitter.train_idxs].nanmean(dim=0)
    if not torch.isfinite(prevalence).all():
        raise ValueError("Cannot impute a descriptor with no observed training labels.")
    binary_targets = torch.where(torch.isfinite(binary_targets), binary_targets, prevalence)
    values = torch.cat([
        binary_targets, datamodule.native_concepts["breast_density"].tensor, diagnosis,
    ], dim=1)
    datamodule.dataset.set_generated_concepts({
        "cbm_targets": AnnotatedTensor(values, annotations.to_concept_space(), axis=1),
    }, use_as_gt=True)
    datamodule.setup("fit")
    train_diagnosis = diagnosis[datamodule.trainset.indices]
    pos_weight = (train_diagnosis == 0).sum() / (train_diagnosis == 1).sum()
    print(f"Training images: {datamodule.train_len}; test images: {datamodule.test_len}")

    # %%
    # Image -> native shape/margin/density concepts -> malignancy.
    backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
    latent_size = backbone.classifier.in_features
    backbone.classifier = torch.nn.Dropout(0.5)
    model = ConceptBottleneckModel(
        input_size=(3, 224, 224), annotations=annotations, task_names=["malignancy"],
        backbone=backbone, latent_size=latent_size,
        lightning=True,
        loss=CompositeLoss([
            ConceptSubset(ConceptLoss(
                binary=torch.nn.BCEWithLogitsLoss(), categorical=torch.nn.CrossEntropyLoss(),
            ), names=concept_names),
            ConceptSubset(ConceptLoss(
                binary=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            ), names=["malignancy"]),
        ], names=["concepts", "malignancy"]),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
        scheduler_class=torch.optim.lr_scheduler.MultiStepLR,
        scheduler_kwargs={"milestones": [31, 50], "gamma": 0.5},
    )

    # %%
    # Fit both heads jointly using the preprocessed targets.
    trainer = Trainer(max_epochs=55, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=datamodule)

    # %%
    # Test: supply only images, and compute AUC over the complete test split.
    model.to(device).eval()
    score_names = binary_names + ["malignancy"]
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            output = model(query=score_names, input=batch["inputs"]["x"].to(device))
            predictions.append(output.logits[score_names].tensor.cpu())
            targets.append(torch.cat([
                batch["concepts"]["native"][binary_names].tensor,
                batch["concepts"]["c"]["malignancy"].tensor,
            ], dim=1))
    scores = torch.cat(predictions).numpy()
    truth = torch.cat(targets).numpy()
    aucs = []
    for index, name in enumerate(score_names):
        known = np.isfinite(truth[:, index])
        auc = roc_auc_score(truth[known, index], scores[known, index])
        aucs.append(auc)
        print(f"Test ROC AUC of {name}: {auc:.4f}")
    print(f"Test concept macro ROC AUC: {np.mean(aucs[:-1]):.4f}")


if __name__ == "__main__":
    main()
