"""
CBIS-DDSM: training a concept bottleneck model
=============================================

Load the official mass training/test splits and train an ImageNet-pretrained
DenseNet121 CBM using native shape, margin, and breast-density labels. The task
is binary malignancy, derived from the dataset's original pathology states.
Train on every training image and report ROC AUC on every test image.

Data and pretrained weights download on first use. Run with:
``python -u examples/data/07_cbis_ddsm.py``.
"""

from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

from torch_concepts import AnnotatedTensor, Annotations, seed_everything
from torch_concepts.data import CBISDDSMDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss


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


class MammogramImages(Dataset):
    """Keep only paths and plain labels in workers to avoid annotation pickling."""

    def __init__(self, dataset, concept_names, augment=False):
        self.paths = [str(Path(dataset.root_dir) / path) for path in dataset.input_data]
        pathology = dataset.native_concepts["pathology"].tensor
        malignant = dataset.annotations.concept("pathology").states.index("MALIGNANT")
        diagnosis = torch.where(torch.isfinite(pathology), (pathology == malignant).float(), torch.nan)
        self.targets = torch.cat([dataset.native_concepts[concept_names].tensor, diagnosis], dim=1)
        augmentation = [
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ] if augment else []
        self.transform = transforms.Compose(augmentation + [
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        dicom = pydicom.dcmread(self.paths[index])
        pixels = dicom.pixel_array.astype(np.float32)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            minimum = -(2 ** (dicom.BitsStored - 1)) if dicom.PixelRepresentation else 0
            pixels = 2 * minimum + 2 ** dicom.BitsStored - 1 - pixels
        image = resize_crop(torch.from_numpy(pixels).unsqueeze(0))
        return self.transform(image), self.targets[index]


class MaskedBCEWithLogitsLoss(torch.nn.Module):
    """Exclude unknown shape/margin labels from concept supervision."""

    def forward(self, input, target):
        known = torch.isfinite(target)
        loss = F.binary_cross_entropy_with_logits(input, target.nan_to_num(), reduction="none")
        return (loss * known).sum() / known.sum().clamp_min(1)


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 55

    # %%
    # Data: preserve official split membership and native concept definitions.
    train, test = [CBISDDSMDataset(
        root=str(DATA_ROOT / "cbis_ddsm"), abnormality_type="mass",
        split=split, image_type="cropped",
    ) for split in ("train", "test")]
    descriptors = train.annotations.resolve(["mass_shape", "mass_margin"])[1]
    binary_names = [name for name in descriptors.labels if not name.endswith(".N/A")]
    concept_names = binary_names + ["breast_density"]
    annotations = train.annotations.subset(concept_names).union_with(
        Annotations(labels=["malignancy"], types=["binary"], states=[["present"]]),
    )
    train_images = MammogramImages(train, concept_names, augment=True)
    test_images = MammogramImages(test, concept_names)
    train_loader = DataLoader(train_images, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_images, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)
    print(f"Training images: {len(train_images)}; test images: {len(test_images)}")

    # %%
    # Image -> native shape/margin/density concepts -> malignancy.
    backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
    latent_size = backbone.classifier.in_features
    backbone.classifier = torch.nn.Dropout(0.5)
    model = ConceptBottleneckModel(
        input_size=(3, 224, 224), annotations=annotations, task_names=["malignancy"],
        backbone=backbone, latent_size=latent_size,
    ).to(device)
    concept_loss = ConceptLoss(
        binary=MaskedBCEWithLogitsLoss(), categorical=torch.nn.CrossEntropyLoss(),
    )
    diagnosis = train_images.targets[:, -1]
    pos_weight = ((diagnosis == 0).sum() / (diagnosis == 1).sum()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[31, 50], gamma=0.5)

    # %%
    # Fit both heads jointly using the native CSV labels.
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, values in train_loader:
            targets = AnnotatedTensor(values.to(device), annotations.to_concept_space(), axis=1)
            optimizer.zero_grad()
            output = model(query=annotations.labels, input=images.to(device))
            loss = concept_loss(output, targets[concept_names]) + F.binary_cross_entropy_with_logits(
                output.logits["malignancy"].tensor, targets["malignancy"].tensor,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}: loss={total_loss / len(train_images):.4f}", flush=True)

    # %%
    # Test: supply only images, and compute AUC over the complete test split.
    model.eval()
    score_names = binary_names + ["malignancy"]
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            output = model(query=score_names, input=images.to(device))
            predictions.append(output.logits[score_names].tensor.cpu())
    scores = torch.cat(predictions).numpy()
    truth = torch.cat([
        test_images.targets[:, :len(binary_names)], test_images.targets[:, -1:],
    ], dim=1).numpy()
    aucs = []
    for index, name in enumerate(score_names):
        known = np.isfinite(truth[:, index])
        auc = roc_auc_score(truth[known, index], scores[known, index])
        aucs.append(auc)
        print(f"Test ROC AUC of {name}: {auc:.4f}")
    print(f"Test concept macro ROC AUC: {np.mean(aucs[:-1]):.4f}")


if __name__ == "__main__":
    main()
