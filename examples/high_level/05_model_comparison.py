"""
Comparing Models
================

High-level models share one interface, so swapping one for another is a
one-line change. Three models are trained the same way here:

- ``BlackBoxTaskOnly``, which predicts the task directly from the input;
- ``ConceptBottleneckModel``, which predicts the task from the concepts only;
- ``ConceptEmbeddingModel``, which predicts it from embeddings of the concepts.

They reach a similar task accuracy, but only the two concept-based models also
tell which concepts they predicted, and can be corrected through them.

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
    BlackBoxTaskOnly,
    ConceptBottleneckModel,
    ConceptEmbeddingModel,
    ConceptLoss,
    GroundTruthIntervention,
    UniformPolicy,
    intervention,
)

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
test = datamodule.testset.indices
x_test, c_test = datamodule.dataset.input_data[test], datamodule.dataset.concepts[test]
one_hot_digit = torch.nn.functional.one_hot(c_test["digit"].flatten().long(), 10)
true_digit = GroundTruthIntervention(torch.logit(one_hot_digit.float(), eps=1e-6))


def parity_accuracy(out):
    return ((out.logits["parity"] > 0).float() == c_test["parity"]).float().mean()


def digit_accuracy(out):
    predicted = out.logits["digit"].argmax(-1, keepdim=True)
    return (predicted == c_test["digit"]).float().mean()


# %%
# Models
# ------
# The same arguments for every model; the CEM also takes the embedding size.
models = {
    "black box": (BlackBoxTaskOnly, {}),
    "CBM": (ConceptBottleneckModel, {}),
    "CEM": (ConceptEmbeddingModel, {"embedding_size": 16}),
}
for name, (model_class, extra) in models.items():
    seed_everything(42)
    model = model_class(
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
        **extra,
    )
    trainer = Trainer(
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluation: the concept-based models also report the digit and accept
    # an intervention on it.
    model.eval()
    with torch.no_grad():
        parity = parity_accuracy(model(query=["parity"], input=x_test))
        line = f"{name:<10} parity accuracy {parity:.3f}"
        if name != "black box":
            digit = digit_accuracy(model(query=["digit"], input=x_test))
            with intervention(model, true_digit, UniformPolicy(), ["digit"]):
                fixed = parity_accuracy(model(query=["parity"], input=x_test))
            line += f" | digit accuracy {digit:.3f}"
            line += f" | parity with the true digit {fixed:.3f}"
    print(line)
