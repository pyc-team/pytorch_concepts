"""
Label-Free Concept Bottleneck Models
====================================

LF-CBM learns a bottleneck without human concept annotations. A generation
pipeline supplies candidate concepts and raw CLIP similarities. ``LabelFreeCBM``
learns a bias-free projection, removes concepts with poor validation alignment,
standardizes learned activations, and fits an elastic-net classifier with SAGA.
The high-level model supports the standard concept/task query interface and
inherits ``ConceptBottleneckModel`` and constructs its own prediction graph.

This example uses the paper's CIFAR-10 training configuration: a frozen OpenAI
CLIP RN50 backbone, all 50000 training images, and the published vocabulary by
default. CLIP ViT-B/16 supplies concept scores. The projection and sparse task
head use the settings recorded in ``lfcbm_cifar10.json``.

Following the authors' code, the 10000 official test images are also used for
validation and projection selection. The reported test accuracy therefore
matches that evaluation protocol; it is not an independent held-out estimate.
The paper reports 86.40% +/- 0.06% over three runs. One run here need not recover
that accuracy or the reported 128 retained concepts. The annotator uses converted
Hugging Face CLIP weights; exact numerical parity has not been established.

Unlike ``data/05_label_free_concepts.py``, raw CLIP similarities are used only
to learn the projection. They are neither sigmoid-calibrated nor treated as
human concept labels. At inference, only the learned concepts and task head run.

Run from the repository root:

.. code-block:: bash

   python -m pip install -e ".[data,lfcbm]"
   python -m pip install git+https://github.com/openai/CLIP.git
   python -m examples.high_level.11_label_free_cbm

To generate a new vocabulary with an LLM, configure your provider's credentials
and run:

.. code-block:: bash

   python -m pip install sentence-transformers
   python -m examples.high_level.11_label_free_cbm --llm-model

Without a value, ``--llm-model`` uses the model configured below. Supply
``--llm-model PROVIDER/MODEL`` to use another LiteLLM model, including local
providers. API credentials and usage limits depend on the selected provider.
The LLM receives class names, using the paper's three few-shot prompts once
per class (30 requests for CIFAR-10). Semantic filtering uses CLIP ViT-B/16 and
all-mpnet-base-v2 before image annotation. New generations can differ from the
published vocabulary; this is not an exact reproduction of its GPT-3 outputs.

Requires the PyC data and lfcbm extras and OpenAI CLIP. CIFAR-10 (~170 MB), the
concept list, and both CLIP checkpoints are downloaded on first run. Extracting
features over all 60000 images can take substantial time
on CPU. The script uses CUDA when available and CPU otherwise. The default
published vocabulary needs no API key. The fitted model is saved to
``~/.cache/pyc/lfcbm/fitted/`` by default; set ``PYC_CACHE`` to change the cache
location.

Reference: Oikarinen et al., ICLR 2023, https://arxiv.org/abs/2304.06129.
"""

# %%
import argparse

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from torch_concepts import Annotations, ImageBackbone, seed_everything
from torch_concepts.data.base import ConceptDataset, ConceptDataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.filters import SemanticConceptFilter
from torch_concepts.data.generation.generators import (
    FixedConceptGenerator, LiteLLMBackend, LLMConceptGenerator,
)
from torch_concepts.data.io import download_url
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import LabelFreeCBM

DEFAULT_LLM_MODEL = "openrouter/nvidia/nemotron-3.5-lightning:free"

parser = argparse.ArgumentParser(description="Train an LF-CBM with the paper's CIFAR-10 settings.")
parser.add_argument(
    "--llm-model", nargs="?", const=DEFAULT_LLM_MODEL,
    metavar="PROVIDER/MODEL",
    help="Generate concepts with the configured LLM, or supply a LiteLLM model identifier",
)
args = parser.parse_args()

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
root = DATA_ROOT / "lfcbm"

# %%
# Task labels only
# ----------------
train = CIFAR10(str(root), train=True, download=True)
test = CIFAR10(str(root), train=False, download=True)
images = torch.from_numpy(np.concatenate([train.data, test.data]))
inputs = images.permute(0, 3, 1, 2).float() / 255
targets = torch.tensor(train.targets + test.targets)
n_train = len(train.data)
dataset = ConceptDataset(
    inputs, targets[:, None],
    annotations=Annotations(labels=["class"], states=[train.classes], types=["categorical"]),
)
dm = ConceptDataModule(
    dataset,
    splitter=FixedIndicesSplitter(range(n_train), range(n_train, len(dataset)), []),
    batch_size=512,
)
dm.setup("fit")

# %%
# Concepts and raw CLIP pseudo-supervision
# ---------------------------------------
def concept_prompts(dataset=None, class_names=None, **kwargs):
    """LF-CBM few-shot prompts for features, surroundings, and superclasses.

    Returns three prompts per class, once each, independently of LLM provider.
    Templates follow the authors' ``GPT_initial_concepts.ipynb`` in
    https://github.com/Trustworthy-ML-Lab/Label-free-CBM.
    """
    templates = (
        'List the most important features for recognizing something as a "goldfish":\n'
        '-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\n'
        'List the most important features for recognizing something as a "beerglass":\n'
        '-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\n'
        'List the most important features for recognizing something as a "{}":',
        'List the things most commonly seen around a "tench":\n'
        '-a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\n'
        'List the things most commonly seen around a "beer glass":\n'
        '-beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\n'
        'List the things most commonly seen around a "{}":',
        'Give superclasses for the word "tench":\n-fish\n-vertebrate\n-animal\n\n'
        'Give superclasses for the word "beer glass":\n-glass\n-container\n-object\n\n'
        'Give superclasses for the word "{}":',
    )
    return [template.format(name) for name in class_names for template in templates]


annotator = CLIPAnnotator(
    model_name="openai/clip-vit-base-patch16", batch_size=512, device=device,
    prompt_template="{}", show_progress=True,
)
if args.llm_model:
    import litellm
    from sentence_transformers import SentenceTransformer

    litellm.suppress_debug_info = True
    completion_kwargs = {"temperature": 0.7, "max_tokens": 1024}
    if args.llm_model == DEFAULT_LLM_MODEL:
        completion_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    generator = LLMConceptGenerator(
        llm=LiteLLMBackend(
            model=args.llm_model,
            system_prompt=(
                "Return only concept names, one per line, without explanations. "
                "Each concept name must be at most 30 characters long."
            ),
            **completion_kwargs,
        ),
        prompt=concept_prompts,
    )
    sentence_encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
    generator_filter = SemanticConceptFilter(
        train.classes, annotator.encode_texts, sentence_encoder.encode,
    )
else:
    reference = "fa2869d46c454f406af69b19d3d5427f2d427492"
    vocabulary_path = download_url(
        f"https://raw.githubusercontent.com/Trustworthy-ML-Lab/Label-free-CBM/{reference}/data/concept_sets/cifar10_filtered.txt",
        str(root), filename="cifar10_filtered.txt",
    )
    with open(vocabulary_path) as file:
        concepts = [line.strip() for line in file if line.strip()]
    generator = FixedConceptGenerator(concepts, cardinalities=[1] * len(concepts))
    generator_filter = None  # The published vocabulary is already filtered.

pipeline = ConceptGenerationPipeline(
    generators=generator, annotators=annotator, generator_filter=generator_filter,
)

# %%
# Multi-stage fitting
# -------------------
lfcbm = LabelFreeCBM(
    pipeline=pipeline,
    backbone=ImageBackbone("RN50", source="openai_clip", device=device),
    clip_cutoff=0.25, interpretability_cutoff=0.45,
    projection_kwargs={"steps": 1000, "batch_size": 50000, "lr": 0.001, "validation_interval": 50},
    classifier_kwargs={"epochs": 1000, "batch_size": 256, "lr": 0.1, "alpha": 0.99,
                       "lam": 0.0007, "tolerance": 1e-4},
    annotation_source="CLIPAnnotator", device=device, seed=42,
)
lfcbm.fit(dm, task_name="class")
print(f"Retained {len(lfcbm.concept_names_)} concepts")
print(lfcbm.concept_names_[:10])
print(lfcbm.diagnostics_["alignment_scores"][lfcbm.concept_names_[:10]])
print(lfcbm.diagnostics_["classifier"])

# %%
# Classification accuracy under the paper's validation/test protocol
# -----------------------------------------------------------------
lfcbm.eval()
test_indices = dm.valset.indices
correct = 0
for start in range(0, len(test_indices), dm.batch_size):
    indices = test_indices[start:start + dm.batch_size]
    predictions = lfcbm.predict(dataset.input_data[indices]).argmax(1).cpu()
    correct += (predictions == targets[indices]).sum().item()
print(f"CIFAR-10 test accuracy (also used for validation): {correct / len(test_indices):.2%} ({correct}/{len(test_indices)})")

# %%
# Learned concepts and task predictions
# ------------------------------------
x = dataset.input_data[n_train:n_train + 8]
activations = lfcbm.predict_concepts(x)  # standardized real-valued projections
logits = lfcbm.predict(x)               # one logit per task class
output = lfcbm(query=["class"], input=x.to(device))
torch.testing.assert_close(output.logits["class"].tensor, logits)
print(activations[lfcbm.concept_names_[:3]])
print([train.classes[index] for index in logits.argmax(1).tolist()])
model = lfcbm                          # itself a high-level ConceptBottleneckModel
torch.testing.assert_close(model(query=["class"], input=x.to(device)).logits["class"].tensor, logits)

# %%
# Persistence without the generation pipeline
# -------------------------------------------
checkpoint = root / "fitted"
lfcbm.save(checkpoint)
# A custom backbone is reconstructed explicitly; the checkpoint supplies weights.
# Reusing this already-created architecture avoids any further model downloads.
restored = LabelFreeCBM.load(checkpoint, backbone=lfcbm.backbone, device=device)
torch.testing.assert_close(restored.predict(x), logits)
torch.testing.assert_close(restored.predict_concepts(x).tensor, activations.tensor)
