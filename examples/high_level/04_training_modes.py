"""
Training Modes
==============

How a concept bottleneck model is trained is chosen by its inference engines:
``train_inference`` is used in training and ``inference`` in evaluation. The
regimes differ in what the task predictor sees while it learns:

- *joint* (the default, ``DeterministicInference``): the predicted concept
  probabilities;
- *independent* (``IndependentInference``): the true concepts, i.e. what an
  expert's intervention will give it;
- *hard*: concepts sampled as exact one-hot values (``AncestralSamplingInference``
  with straight-through distributions, set through ``variable_distributions``),
  so that it never sees the model's confidence; predictions then use each
  concept's most likely value (``MAPForwardInference``).

Here the digit determines the parity, so all three end up equally accurate,
and all three are fixed by revealing the true digit.

Data: Color-MNIST, as in ``01_concept_bottleneck_model.py``.
"""

# %%
import torch
from pyro.distributions import (
    RelaxedBernoulliStraightThrough,
    RelaxedOneHotCategoricalStraightThrough,
)
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    AncestralSamplingInference,
    ConceptBottleneckModel,
    ConceptLoss,
    GroundTruthIntervention,
    IndependentInference,
    MAPForwardInference,
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


# %%
# Training modes
# --------------
modes = {
    "joint": {},
    "independent": {"train_inference": IndependentInference},
    "hard": {
        "variable_distributions": {
            "binary": RelaxedBernoulliStraightThrough,
            "categorical": RelaxedOneHotCategoricalStraightThrough,
        },
        "train_inference": AncestralSamplingInference,
        "inference": MAPForwardInference,
    },
}
for name, mode in modes.items():
    seed_everything(42)
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
        **mode,
    )
    trainer = Trainer(
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=datamodule)

    # ``model.eval()`` switches to the evaluation engine
    model.eval()
    with torch.no_grad():
        out = model(query=["parity"], input=x_test)
        with intervention(model, true_digit, UniformPolicy(), ["digit"]):
            out_true_digit = model(query=["parity"], input=x_test)
    before, after = parity_accuracy(out), parity_accuracy(out_true_digit)
    print(f"{name:<12} parity accuracy {before:.3f} | with the true digit {after:.3f}")
