"""
Composing Losses
================

``ConceptLoss`` gives each concept type its own loss. Losses compose further:

- several terms on one type, as a weighted list (here an L1 penalty on the
  logits of the categorical concepts);
- terms on named groups of concepts: ``ConceptSubset`` restricts a loss to some
  concepts and ``CompositeLoss`` sums weighted terms. ``WeightedConceptLoss``
  builds the common case of concepts and tasks weighted separately;
- custom terms that read the whole model output, as subclasses of ``PyCLoss``.

``CompositeLoss.breakdown`` returns each weighted term, to see what drives the
total. Any of these losses is passed to a model as ``loss=...`` or called as
``loss(output, target)`` in a plain training loop.

Data: one batch of Color-MNIST, as in ``01_concept_bottleneck_model.py``.
"""

# %%
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    CompositeLoss,
    ConceptBottleneckModel,
    ConceptLoss,
    ConceptSubset,
    L1LogitRegularizer,
    PyCLoss,
    WeightedConceptLoss,
)

seed_everything(42)

# %%
# A model output to score
# -----------------------
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    max_samples=10000,
    batch_size=256,
)
datamodule.setup()
batch = next(iter(datamodule.train_dataloader()))
model = ConceptBottleneckModel(
    input_size=datamodule.n_features,
    annotations=datamodule.annotations,
    task_names=["parity"],
    backbone=torch.nn.Sequential(torch.nn.Flatten(), MLP(3 * 28 * 28, 128)),
    latent_size=128,
)
out = model(query=["digit", "color", "parity"], input=batch["inputs"]["x"])
target = batch["concepts"]["c"]

# %%
# One loss per type
# -----------------
loss = ConceptLoss(binary=BCEWithLogitsLoss(), categorical=CrossEntropyLoss())
print(f"{loss}: {loss(out, target):.4f}")

# Several terms on one type, with weights.
loss = ConceptLoss(
    binary=BCEWithLogitsLoss(),
    categorical=[CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
    categorical_weights=[1.0, 0.5],
)
print(f"{loss}: {loss(out, target):.4f}")

# %%
# Concepts and task weighted separately
# -------------------------------------
loss = WeightedConceptLoss(
    concept_weight=0.5,
    task_weight=1.0,
    task_names=["parity"],
    binary=BCEWithLogitsLoss(),
    categorical=CrossEntropyLoss(),
)
terms = loss.breakdown(out, target)
print({name: round(term.item(), 4) for name, term in terms.items()})

# The same loss, written out: one subset of concepts per term.
loss = CompositeLoss(
    terms=[
        ConceptSubset(
            ConceptLoss(binary=BCEWithLogitsLoss(), categorical=CrossEntropyLoss()),
            exclude=["parity"],
        ),
        ConceptSubset(
            ConceptLoss(binary=BCEWithLogitsLoss(), categorical=CrossEntropyLoss()),
            names=["parity"],
        ),
    ],
    weights=[0.5, 1.0],
    names=["concepts", "tasks"],
)
terms = loss.breakdown(out, target)
print({name: round(term.item(), 4) for name, term in terms.items()})


# %%
# A custom term
# -------------
# A ``PyCLoss`` receives the whole model output (and, if needed, the target and
# the model). This one penalizes every logit the model reports, of any type.
class LogitL1(PyCLoss):
    def forward(self, input, target=None, model=None):
        return input.logits.tensor.abs().mean()


loss = CompositeLoss(
    terms=[
        ConceptLoss(binary=BCEWithLogitsLoss(), categorical=CrossEntropyLoss()),
        LogitL1(),
    ],
    weights=[1.0, 0.01],
    names=["supervision", "logit_l1"],
)
terms = loss.breakdown(out, target)
print({name: round(term.item(), 4) for name, term in terms.items()})
