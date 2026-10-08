"""
Concept Bottleneck Model
========================

High-level models come assembled. A ``ConceptBottleneckModel`` needs the input
size, the concept annotations, the names of the tasks among them, and a
backbone; it is a plain ``torch.nn.Module``, trained here with an ordinary
PyTorch loop (``02_lightning_training.py`` uses Lightning instead).

Data: Color-MNIST, MNIST digits colored red or green. The model predicts two
concepts, ``digit`` (10 classes) and ``color`` (2 classes), and from them the
task ``parity`` (1 if the digit is even). MNIST (~60 MB) is downloaded on
first run.

Reference: Koh et al., "Concept Bottleneck Models", ICML 2020.
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import MLP, ConceptBottleneckModel, ConceptLoss

seed_everything(42)

# %%
# Data
# ----
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    max_samples=10000,
    batch_size=256,
)
datamodule.setup()
print(datamodule.annotations.labels, datamodule.annotations.cardinalities)

# %%
# Model
# -----
model = ConceptBottleneckModel(
    input_size=datamodule.n_features,  # (3, 28, 28) images
    annotations=datamodule.annotations,
    task_names=["parity"],
    backbone=torch.nn.Sequential(torch.nn.Flatten(), MLP(3 * 28 * 28, 128)),
    latent_size=128,
)

# %%
# Training
# --------
# ``query`` lists the variables to compute; the output holds their logits by
# name. ``ConceptLoss`` applies one loss per concept type.
query = ["digit", "color", "parity"]
loss_fn = ConceptLoss(
    binary=torch.nn.BCEWithLogitsLoss(),
    categorical=torch.nn.CrossEntropyLoss(),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(10):
    model.train()
    for batch in datamodule.train_dataloader():
        optimizer.zero_grad()
        out = model(query=query, input=batch["inputs"]["x"])
        loss = loss_fn(out, batch["concepts"]["c"])
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        print(f"epoch {epoch} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
# Labels hold one column per concept: 0/1 for a binary concept, the class
# index for a categorical one.
test = datamodule.testset.indices
x_test, c_test = datamodule.dataset.input_data[test], datamodule.dataset.concepts[test]


def accuracy(logits, labels):
    if logits.shape[-1] == 1:  # a binary concept has a single logit
        predicted = (logits > 0).float()
    else:
        predicted = logits.argmax(-1, keepdim=True).float()
    return (predicted == labels).float().mean().item()


model.eval()
with torch.no_grad():
    out = model(query=query, input=x_test)
for name in query:
    print(f"test accuracy of {name}: {accuracy(out.logits[name], c_test[name]):.3f}")
