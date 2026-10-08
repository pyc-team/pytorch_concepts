"""
Label-Free Concepts
===================

When a dataset has no concept labels, they can be generated:

1. an LLM proposes a vocabulary of visual concepts, prompted with a few labeled
   training images (``LLMConceptGenerator`` with ``LiteLLMBackend``);
2. CLIP scores every image against every concept (``CLIPAnnotator``);
3. the scores are calibrated into probabilities and uncertain annotations are
   filtered out (``SigmoidCalibrator``, ``ThresholdAnnotationFilter``).

``ConceptGenerationPipeline`` chains the three steps, and
``datamodule.generate_concepts`` attaches its output to the dataset, next to
the native concepts. Here the vocabulary is generated from the training images
only and annotated on all of them; the pipeline also accepts index lists or
separate datasets per split. A concept bottleneck model is then trained to
predict the native ``parity`` of Color-MNIST digits from the generated
concepts.

Requires an API key for the LLM provider, e.g. ``export GEMINI_API_KEY=...``
(``OPENAI_API_KEY`` for an ``openai/...`` model). MNIST (~60 MB) and CLIP are
downloaded on first run.

References: Oikarinen et al., "Label-Free Concept Bottleneck Models", ICLR
2023; Yang et al., "Language in a Bottle: Language Model Guided Concept
Bottlenecks for Interpretable Image Classification", CVPR 2023.
"""

# %%
import base64
import math
from io import BytesIO

import torch
from PIL import Image
from pytorch_lightning import Trainer
from torch import nn
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.filters import ThresholdAnnotationFilter
from torch_concepts.data.generation.generators import (
    LiteLLMBackend,
    LLMConceptGenerator,
)
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    ConceptBottleneckModel,
    ConceptLoss,
    ConceptMetrics,
    IndependentInference,
)

seed_everything(0)

# Get Ollamaa from https://ollama.com/
# > ollama pull gemma4 (or your favourite LiteLLM model)
# https://getdeploying.com/guides/local-gemma4
LLM = "ollama_chat/gemma4:e4b"  # any LiteLLM model id

# %%
# Data
# ----
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    coloring={"red": range(6), "green": range(6, 10)},
    batch_size=128,
    max_samples=10000,
    seed=0,
)
datamodule.setup("fit")
dataset = datamodule.dataset


# %%
# The prompt
# ----------
# A few training images with their digit and color go to the LLM, which expands
# them into a broader vocabulary. ``indices`` are the rows the pipeline allows
# for concept discovery: here, the training split. Concepts that name the task
# (parity, even, odd) are ruled out: they would hand the answer to the model.
def image_url(image):
    """Encode a (3, H, W) image for a multimodal prompt."""
    array = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prompt(dataset, class_names=None, indices=None, **kwargs):
    instruction = (
        "Expand the partial digit and color concept evidence in the labeled images "
        "below into 12 short visual binary concepts useful for classifying ColorMNIST "
        f"images as {class_names}. Include digit identity, color, and broader "
        "simple shape properties. Do not name the classes: no concept may mention "
        "parity, even or odd. Return one concept per line and no explanations."
    )
    content = [{"type": "text", "text": instruction}]
    positions = torch.linspace(0, len(indices) - 1, steps=4).long()
    native = dataset.native_concepts.annotations.labels
    for position in positions.tolist():
        sample = dataset[int(indices[position])]
        concepts = sample["concepts"]["native"]
        digit = int(concepts[native.index("digit")])
        color = ("red", "green")[int(concepts[native.index("color")])]
        url = image_url(sample["inputs"]["x"])
        content += [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": f"Example label: {color} digit {digit}."},
        ]
    return [{"role": "user", "content": content}]


# %%
# Generating the concepts
# -----------------------
# CLIP's similarities fall in a narrow range, with a different baseline for each
# concept. ``standardize=True`` rescales each concept's scores to zero mean and
# unit variance before the sigmoid, so the filter keeps above-average matches.
pipeline = ConceptGenerationPipeline(
    generators=LLMConceptGenerator(
        llm=LiteLLMBackend(model=LLM, temperature=0.0, timeout=120.0),
        prompt=prompt,
    ),
    annotators=CLIPAnnotator(
        model_name="openai/clip-vit-base-patch32",
        prompt_template="a photo of a {}",
        batch_size=128,
        show_progress=True,
    ),
    calibrator=SigmoidCalibrator(standardize=True),
    calibrated_annotation_filter=ThresholdAnnotationFilter(0.5),
    routing="merged",  # one concept axis for all generated concepts
)
datamodule.generate_concepts(
    pipeline,
    generation_indices=datamodule.trainset.indices,
    use_as_gt=False,  # the training target is set below
    class_names=["even", "odd"],
)
generated = dataset.generated_concepts["CLIPAnnotator"]
print("generated concepts:", generated.annotations.labels)

# %%
# The training target
# -------------------
# A batch holds the concepts in ``batch["concepts"]``, under three keys:
#
# - ``"native"``: the dataset's own labels (digit, color, parity);
# - ``"generated"``: one entry per pipeline output;
# - ``"c"``: the target that models train on, a single source: the native
#   concepts by default, or a generated one selected with ``use_as_gt=True``.
#
# Here we want to play with a model that uses both: the generated concepts as its
# concepts and the native ``parity`` as its task. So the two are joined into one source,
# stored with ``set_generated_concepts`` (which replaces the generated sources)
# and selected as the target. The datamodule reads its batches and its
# ``annotations`` from the dataset, so nothing needs rebuilding.
parity = dataset.native_concepts[["parity"]]
dataset.set_generated_concepts(
    {"clip_and_parity": generated.union_with(parity)},
    use_as_gt=True,
)
print("training target:", datamodule.annotations.labels)

# %%
# A model on the generated concepts
# ---------------------------------
model = ConceptBottleneckModel(
    input_size=datamodule.n_features,
    annotations=datamodule.annotations,
    task_names=["parity"],
    backbone=nn.Flatten(),
    latent_size=math.prod(datamodule.n_features),
    train_inference=IndependentInference,
    lightning=True,
    loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
    metrics=ConceptMetrics(
        annotations=datamodule.annotations,
        binary={"accuracy": BinaryAccuracy},
        summary=False,  # parity only: the generated labels are soft, not 0/1
        per_concept=["parity"],
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 0.01, "weight_decay": 1e-3},
)
trainer = Trainer(max_epochs=5, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
