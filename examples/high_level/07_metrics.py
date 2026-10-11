"""
Metrics
=======

``ConceptMetrics`` computes torchmetrics metrics by concept type, for every
concept (``per_concept=True``) and as a summary per type (``summary=True``). A
metric is given in one of three ways:

- an instance, used as is for all concepts of the type;
- a class, built once per concept, so that a multiclass metric gets each
  concept's number of classes;
- a ``(class, kwargs)`` tuple: the same, with extra arguments.

The metrics below are computed outside of Lightning, on the test predictions
of a model trained as in ``02_lightning_training.py``.

Data: Color-MNIST, as in ``01_concept_bottleneck_model.py``.
"""

# %%
import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassF1Score,
)

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import MLP, ConceptBottleneckModel, ConceptLoss, ConceptMetrics

seed_everything(42)

# %%
# A trained model
# ---------------
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    max_samples=10000,
    batch_size=256,
)
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
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 1e-3},
)
trainer = Trainer(
    max_epochs=10,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(model, datamodule=datamodule)

test = datamodule.testset.indices
x_test, c_test = datamodule.dataset.input_data[test], datamodule.dataset.concepts[test]
model.eval()
with torch.no_grad():
    out = model(query=["digit", "color", "parity"], input=x_test)

# %%
# Metrics
# -------
metrics = ConceptMetrics(
    annotations=datamodule.annotations,
    binary={
        "accuracy": BinaryAccuracy(),  # an instance
        "f1": BinaryF1Score(),
    },
    categorical={
        "accuracy": MulticlassAccuracy,  # a class
        "macro_f1": (MulticlassF1Score, {"average": "macro"}),  # a class with kwargs
    },
    per_concept=True,
    summary=True,
)
metrics.update(out, c_test)
for name, value in sorted(metrics.compute().items()):
    print(f"{name:<28} {value:.3f}")
