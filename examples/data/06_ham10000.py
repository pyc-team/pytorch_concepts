"""
HAM10000
========

HAM10000 holds dermoscopic images of skin lesions, shipped with four native
concepts: ``diagnosis``, ``age``, ``sex`` and ``localization``.
``HAM10000DataModule`` loads them like any other PyC dataset.

Concepts can also be generated from the images with the smallest possible
pipeline: ``FixedConceptGenerator`` returns a hand-written vocabulary,
``CLIPAnnotator`` scores every image against every concept with a frozen CLIP,
and ``SigmoidCalibrator`` turns the raw cosine similarities into probabilities.
Two implausible concepts, a dog and a cat, are added as a sanity check.

Data: 256 HAM10000 images (the image archives, a few GB, are downloaded from
Harvard Dataverse on first run). For an LLM-generated vocabulary, see
``data/05_label_free_concepts.py``.
"""

# %%
from torch_concepts import seed_everything
from torch_concepts.data import HAM10000DataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.generators import FixedConceptGenerator
from torch_concepts.env import DATA_ROOT

seed_everything(42)

# %%
# Data
# ----
# HAM10000 ships no official split, so the datamodule draws a random one.
dm = HAM10000DataModule(
    root=str(DATA_ROOT / "ham10000"),
    batch_size=64,
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
# Dermoscopic criteria written by hand, plus two concepts that should score low.
CONCEPTS = [
    "dermoscopy of an asymmetric lesion",
    "dermoscopy of an irregular border",
    "dermoscopy of a blue-white veil",
    "an image of a dog",
    "an image of a cat",
]

# One generator proposing the vocabulary, one annotator scoring it, and a
# calibrator turning the raw cosine similarities into probabilities.
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
generated = pipeline(dm.dataset)["CLIPAnnotator"]
print(f"Generated: {tuple(generated.shape)}")

for index, label in enumerate(generated.annotation.labels):
    score = generated.tensor[:, index].mean()
    print(f"  Sigmoid score of '{label}' = {score:.2f}")
