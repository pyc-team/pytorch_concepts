"""
SIIM-ACR Pneumothorax
======================

SIIM-ACR contains chest radiographs with a native binary ``pneumothorax``
concept. Downloading requires Kaggle credentials; a community DICOM re-host is
several GB, even when this example uses only 256 samples.

The example also scores a small set of concepts with CXR-CLIP, a CLIP pretrained
on chest radiographs and their reports. Its checkpoint (``r50_mcc.tar``, CC BY-NC
4.0) is downloaded by hand from https://github.com/Soombit-ai/cxr-clip into
``~/.cache/pyc/cxr_clip``. The standardized scores are relative to this example's
samples, not clinical probabilities.
"""
# %%
from torch_concepts import seed_everything
from torch_concepts.data import PneumothoraxDataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CXRCLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.generators import FixedConceptGenerator
from torch_concepts.env import CACHE, DATA_ROOT

seed_everything(42)

# %%
# Data
# ----
dm = PneumothoraxDataModule(
    root=str(DATA_ROOT / "pneumothorax"),
    batch_size=32,
    max_samples=256,
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
# Generated concepts
# ------------------
CONCEPTS = [
    "a chest x-ray with a visible pleural line",
    "a chest x-ray with peripheral lung markings",
    "a chest x-ray with apical hyperlucency",
    "a chest x-ray with a deep sulcus sign",
    "a chest x-ray with mediastinal shift",
]

pipeline = ConceptGenerationPipeline(
    generators=FixedConceptGenerator(
        labels=CONCEPTS,
        types=["binary"] * len(CONCEPTS),
        cardinalities=[1] * len(CONCEPTS),
    ),
    annotators=CXRCLIPAnnotator(
        checkpoint=CACHE / "cxr_clip" / "r50_mcc.tar",
        prompt_template="{}",
        batch_size=8,
        show_progress=True,
    ),
    calibrator=SigmoidCalibrator(standardize=True),
)
generated = pipeline(dm.dataset)["CXRCLIPAnnotator"]
print(f"Generated: {tuple(generated.shape)}")

for index, label in enumerate(generated.annotation.labels):
    scores = generated.tensor[:, index]
    print(f"  '{label}': mean={scores.mean():.2f}, range={scores.min():.2f}-{scores.max():.2f}")
