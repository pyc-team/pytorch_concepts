"""
Example: Composing multiple loss terms with the list-based API

Two scenarios are demonstrated:
  1. Single loss (backwards-compatible)
  2. List of losses with custom weights
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torchmetrics.classification import BinaryAccuracy
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, L1LogitRegularizer
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule


def evaluate(model, datamodule, concept_names, n_concepts):
    """Print test-set concept and task accuracy."""
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()
    model.eval()
    c_acc, t_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            out = model(x=batch['inputs']['x'], query=concept_names)
            c_acc += concept_acc_fn(out[:, :n_concepts], batch['concepts']['c'][:, :n_concepts].int()).item()
            t_acc += task_acc_fn(out[:, n_concepts:], batch['concepts']['c'][:, n_concepts:].int()).item()
            n += 1
    print(f"Concept accuracy: {c_acc / n:.4f}")
    print(f"Task accuracy:    {t_acc / n:.4f}")


def main():
    seed_everything(42)

    # Data
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=10_000)
    datamodule = ConceptDataModule(dataset=dataset, batch_size=2048, val_size=0.1, test_size=0.2)
    annotations = dataset.annotations
    concept_names = annotations.get_axis_annotation(1).labels
    n_features = dataset.input_data.shape[1]
    n_concepts = 2
    variable_distributions = {name: Bernoulli for name in concept_names}

    # Shared model kwargs
    model_kwargs = dict(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        lightning=True,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.02},
    )

    # ── Scenario 1: single loss ──────────────────────────────
    print("=" * 60)
    print("Scenario 1: Single loss")
    print("=" * 60)

    concept_loss = ConceptLoss(
        annotations=annotations,
        binary=nn.BCEWithLogitsLoss()
    )

    model = ConceptBottleneckModel(
        **model_kwargs, 
        loss=concept_loss
    )
    Trainer(max_epochs=50, enable_progress_bar=True).fit(model, datamodule=datamodule)
    evaluate(model, datamodule, concept_names, n_concepts)



    # ── Scenario 2: list of losses with weights ──────────────
    print("\n" + "=" * 60)
    print("Scenario 2: Sum of two losses with weights [1.0, 0.5]")
    print("=" * 60)

    concept_loss = ConceptLoss(
        annotations=annotations,
        binary=nn.BCEWithLogitsLoss()
    )
    reg = L1LogitRegularizer(scale=0.01)

    model = ConceptBottleneckModel(
        **model_kwargs,
        loss=[concept_loss, reg],
        loss_weights=[1.0, 0.5],
    )
    Trainer(max_epochs=50, enable_progress_bar=True).fit(model, datamodule=datamodule)
    evaluate(model, datamodule, concept_names, n_concepts)


if __name__ == "__main__":
    main()
