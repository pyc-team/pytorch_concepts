"""
Training with Lightning
=======================

With ``lightning=True``, a high-level model is also a PyTorch Lightning module:
give it a loss, metrics and an optimizer, and a ``Trainer`` runs training,
validation and testing on a datamodule. ``ConceptLoss`` and ``ConceptMetrics``
route every concept to the loss and metrics of its type, here cross-entropy
and multiclass accuracy for ``digit`` and ``color`` and their binary
counterparts for ``parity``.

Data: Color-MNIST, as in ``01_concept_bottleneck_model.py``.
"""

# %%
import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import MLP, ConceptBottleneckModel, ConceptLoss, ConceptMetrics

seed_everything(42)

# %%
# Data
# ----
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    max_samples=10000,
    batch_size=256,
)

# %%
# Model
# -----
# A metric class (rather than an instance) is built per concept, so that
# ``MulticlassAccuracy`` gets each concept's number of classes.
model = ConceptBottleneckModel(
    input_size=datamodule.n_features,
    annotations=datamodule.annotations,
    task_names=["parity"],
    backbone=torch.nn.Sequential(torch.nn.Flatten(), MLP(3 * 28 * 28, 128)),
    latent_size=128,
    lightning=True,
    loss=ConceptLoss(
        binary=torch.nn.BCEWithLogitsLoss(),
        categorical=torch.nn.CrossEntropyLoss(),
    ),
    metrics=ConceptMetrics(
        annotations=datamodule.annotations,
        binary={"accuracy": BinaryAccuracy},
        categorical={"accuracy": MulticlassAccuracy},
        per_concept=True,
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 1e-3},
)

# %%
# Training and testing
# --------------------
trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
