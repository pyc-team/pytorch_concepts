"""
CBIS-DDSM
=========

CBIS-DDSM holds scanned-film mammograms with native radiological concepts and
three pathology states. Each sample describes an abnormality on one breast/view.
``CBISDDSMDataset`` loads the official training and test CSVs separately.

Like the HAM10000 example, a ``FixedConceptGenerator`` proposes a vocabulary,
``CLIPAnnotator`` scores the images, and ``SigmoidCalibrator`` converts cosine
similarities into probabilities. A dog and a cat are added as sanity checks.

Data: up to 256 training mass crops. The datasets first try to download the
official mass train/test CSVs and missing image series from TCIA. If downloading
fails or the API filenames cannot be mapped uniquely to the CSV, the error asks
for manual acquisition from https://www.cancerimagingarchive.net/collection/cbis-ddsm/.
Put CSVs in ``DATA_ROOT/cbis_ddsm/raw`` and images below ``DATA_ROOT/cbis_ddsm``,
preserving ``StudyUID/SeriesUID/filename.dcm`` suffixes matching the CSV.
Each selected split's missing sources are acquired before the 256-image subset
is selected. CLIP weights are downloaded on first run.

Image scaling, resizing, and RGB replication below are choices for this CLIP
example. The dataset keeps its native DICOM pixels and original annotations.
Reference: Lee et al., Scientific Data 4, 170177 (2017),
https://doi.org/10.1038/sdata.2017.177.
"""

# %%
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torch_concepts import seed_everything
from torch_concepts.data import CBISDDSMDataset
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.generators import FixedConceptGenerator
from torch_concepts.env import DATA_ROOT

seed_everything(42)

# %%
# Images for CLIP
# ---------------
# Scale each crop to [0, 1], resize it for batching, and repeat grayscale as RGB.
# The CLIP processor applies its own model-specific normalization afterwards.
def clip_transform(image):
    image = (image - image.min()) / (image.max() - image.min()).clamp_min(1)
    image = F.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(0)
    return image.repeat(3, 1, 1)


# %%
# Data
# ----
# Membership comes directly from the official CSVs; no random split is drawn.
train_dataset = CBISDDSMDataset(
    root=str(DATA_ROOT / "cbis_ddsm"),
    abnormality_type="mass",
    split="train",
    image_type="cropped",
    transform=clip_transform,
)
test_dataset = CBISDDSMDataset(
    root=str(DATA_ROOT / "cbis_ddsm"),
    abnormality_type="mass",
    split="test",
    image_type="cropped",
    transform=clip_transform,
)
print(f"Official split sizes: {len(train_dataset)} train, {len(test_dataset)} test")
print(f"Concepts: {train_dataset.concept_names}")
print(f"Cardinalities: {train_dataset.annotations.cardinalities}")
print(f"Pathology states: {train_dataset.annotations.concept('pathology').states}")

# Use the first 256 training observations to keep this demonstration small.
train_images = Subset(train_dataset, range(min(256, len(train_dataset))))
loader = DataLoader(train_images, batch_size=64, collate_fn=train_dataset.collate)
batch = next(iter(loader))
print(f"inputs:   {tuple(batch['inputs']['x'].shape)}")
print(f"concepts: {tuple(batch['concepts']['c'].shape)}")
print(train_dataset.metadata[["patient_id", "image_view", "abnormality_id", "pathology"]].head())

# %%
# Generated concepts
# ------------------
CONCEPTS = [
    "a mammogram showing an oval mass",
    "a mammogram showing an irregular mass",
    "a mammogram showing a spiculated mass",
    "an image of a dog",
    "an image of a cat",
]

pipeline = ConceptGenerationPipeline(
    generators=FixedConceptGenerator(
        labels=CONCEPTS,
        types=["binary"] * len(CONCEPTS),
        cardinalities=[1] * len(CONCEPTS),
    ),
    annotators=CLIPAnnotator(
        model_name="openai/clip-vit-large-patch14",
        prompt_template="{}",
        batch_size=64,
        show_progress=True,
    ),
    calibrator=SigmoidCalibrator(scale=100),
)
generated = pipeline(train_images)["CLIPAnnotator"]
print(f"Generated: {tuple(generated.shape)}")

for index, label in enumerate(generated.annotations.labels):
    score = generated.tensor[:, index].mean()
    print(f"  Sigmoid score of '{label}' = {score:.2f}")
