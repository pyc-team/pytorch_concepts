"""
Interventions
=============

``intervention`` works on a high-level model exactly as on a layer: name the
concepts, choose a strategy and a policy, and every forward pass inside the
``with`` block uses the new concept values. Because the task is predicted from
the concepts only, the interventions answer two questions:

- does correcting a concept fix the task? Parity depends on the digit, so
  revealing the true digit should fix the parity mistakes, and revealing the
  true color should change nothing;
- what would the model predict if a concept took a given value? Forcing every
  digit to be a 3 should make every prediction "odd".

Data: Color-MNIST, as in ``01_concept_bottleneck_model.py``.
"""

# %%
import torch
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    ConceptBottleneckModel,
    ConceptLoss,
    DoIntervention,
    GroundTruthIntervention,
    UniformPolicy,
    intervention,
)

seed_everything(42)

# %%
# Data and model
# --------------
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
trainer = Trainer(max_epochs=20, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)

test = datamodule.testset.indices
x_test, c_test = datamodule.dataset.input_data[test], datamodule.dataset.concepts[test]
query = ["digit", "color", "parity"]


def report(title, out):
    predicted_digit = out.logits["digit"].argmax(-1, keepdim=True)
    digit = (predicted_digit == c_test["digit"]).float().mean()
    parity = ((out.logits["parity"] > 0).float() == c_test["parity"]).float().mean()
    print(f"{title:<26} digit accuracy {digit:.3f} | parity accuracy {parity:.3f}")


model.eval()
with torch.no_grad():
    report("no intervention", model(query=query, input=x_test))


# %%
# Ground-truth interventions
# --------------------------
# Concepts are predicted as logits: a true class becomes a large logit, the
# other classes large negative ones.
def true_logits(name, n_classes):
    one_hot = torch.nn.functional.one_hot(c_test[name].flatten().long(), n_classes)
    return torch.logit(one_hot.float(), eps=1e-6)


true_digit = GroundTruthIntervention(true_logits("digit", 10))
true_color = GroundTruthIntervention(true_logits("color", 2))
with torch.no_grad():
    with intervention(model, true_digit, UniformPolicy(), ["digit"]):
        report("true digit", model(query=query, input=x_test))
    with intervention(model, true_color, UniformPolicy(), ["color"]):
        report("true color", model(query=query, input=x_test))

# %%
# Do-interventions
# ----------------
three = torch.full((10,), -10.0)
three[3] = 10.0
with torch.no_grad():
    with intervention(model, DoIntervention(three), UniformPolicy(), ["digit"]):
        out = model(query=query, input=x_test)
odd = (out.logits["parity"] < 0).float().mean()
print(f"do(digit = 3): {odd:.0%} of the images predicted odd")
