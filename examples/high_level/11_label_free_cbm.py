"""
Label-Free Concept Bottleneck Models
====================================

LF-CBM learns a bottleneck without human concept annotations. A generation
pipeline supplies candidate concepts and raw CLIP similarities. ``LabelFreeCBM``
learns a bias-free projection, removes concepts with poor validation alignment,
standardizes learned activations, and fits an elastic-net classifier with SAGA.
The high-level model supports the standard concept/task query interface and
inherits ``ConceptBottleneckModel`` and constructs its own prediction graph.

This example uses CIFAR-10 task labels, the authors' published vocabulary, and
a frozen ImageNet ResNet-18 backbone. The backbone and small data subset make
this a demonstration; the accompanying ``lfcbm_cifar10.json`` gives the original
CLIP RN50 backbone / ViT-B/16 annotation configuration for reproduction.

Unlike ``data/05_label_free_concepts.py``, raw CLIP similarities are used only
to learn the projection. They are neither sigmoid-calibrated nor treated as
human concept labels. At inference, only the learned concepts and task head run.

Run from the repository root:

.. code-block:: bash

   python -m pip install -e ".[data,lfcbm]"
   python -m examples.high_level.11_label_free_cbm

Requires the PyC data and lfcbm extras. CIFAR-10 (~170 MB), the concept list, CLIP, and
ResNet weights are downloaded on first run. Feature extraction can take time
on CPU. The script uses CUDA when available and CPU otherwise. No API key is
needed. The fitted model is saved to ``~/.cache/pyc/lfcbm/fitted/`` by default;
set ``PYC_CACHE`` to change the cache location.

Reference: Oikarinen et al., ICLR 2023, https://arxiv.org/abs/2304.06129.
"""

# %%
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from torch_concepts import Annotations, ImageBackbone, seed_everything
from torch_concepts.data.base import ConceptDataset, ConceptDataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.generators import FixedConceptGenerator
from torch_concepts.data.io import download_url
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import LabelFreeCBM

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
root = DATA_ROOT / "lfcbm"

# %%
# Task labels only
# ----------------
train = CIFAR10(str(root), train=True, download=True)
test = CIFAR10(str(root), train=False, download=True)
images = torch.from_numpy(np.concatenate([train.data[:6000], test.data[:1000]]))
inputs = images.permute(0, 3, 1, 2).float() / 255
targets = torch.tensor(train.targets[:6000] + test.targets[:1000])
dataset = ConceptDataset(
    inputs, targets[:, None],
    annotations=Annotations(labels=["class"], states=[train.classes], types=["categorical"]),
)
dm = ConceptDataModule(
    dataset,
    splitter=FixedIndicesSplitter(range(5000), range(5000, 6000), range(6000, 7000)),
    batch_size=64,
)

# %%
# Concepts and raw CLIP pseudo-supervision
# ---------------------------------------
reference = "fa2869d46c454f406af69b19d3d5427f2d427492"
vocabulary_path = download_url(
    f"https://raw.githubusercontent.com/Trustworthy-ML-Lab/Label-free-CBM/{reference}/data/concept_sets/cifar10_filtered.txt",
    str(root), filename="cifar10_filtered.txt",
)
with open(vocabulary_path) as file:
    concepts = [line.strip() for line in file if line.strip()]
pipeline = ConceptGenerationPipeline(
    generators=FixedConceptGenerator(concepts, cardinalities=[1] * len(concepts)),
    annotators=CLIPAnnotator(
        model_name="openai/clip-vit-base-patch16", batch_size=64, device=device,
        prompt_template="{}", show_progress=True,
    ),
)

# %%
# Multi-stage fitting
# -------------------
lfcbm = LabelFreeCBM(
    pipeline=pipeline,
    backbone=ImageBackbone("resnet18", device=device),
    task_name="class", class_names=train.classes,
    annotation_source="CLIPAnnotator", batch_size=64, device=device, seed=42,
)
lfcbm.fit(dm)
print(f"Retained {len(lfcbm.concept_names_)} concepts")
print(lfcbm.concept_names_[:10])
print(lfcbm.diagnostics_["alignment_scores"][lfcbm.concept_names_[:10]])
print(lfcbm.diagnostics_["classifier"])

# %%
# Classification accuracy on the held-out test subset
# --------------------------------------------------
lfcbm.eval()
test_indices = dm.testset.indices
correct = 0
for start in range(0, len(test_indices), dm.batch_size):
    indices = test_indices[start:start + dm.batch_size]
    predictions = lfcbm.predict(dataset.input_data[indices]).argmax(1).cpu()
    correct += (predictions == targets[indices]).sum().item()
print(f"Held-out test accuracy: {correct / len(test_indices):.2%} ({correct}/{len(test_indices)})")

# %%
# Learned concepts and task predictions
# ------------------------------------
x = dataset.input_data[6000:6008]
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
