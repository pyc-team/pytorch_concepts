"""
Example: Composing multiple loss terms with per-type weights

Uses the **insurance** Bayesian-network dataset which has **both** binary
and categorical concepts, making it a good test-bed for type-specific loss
composition.

Three scenarios are demonstrated:
  1. Single loss per type (backward-compatible)
  2. Per-type composite losses with custom weights
  3. Composite only on binary concepts, plain loss on categorical
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, OneHotCategorical
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, L1LogitRegularizer
from torch_concepts.data.datamodules import BnLearnDataModule


def main():
    seed_everything(42)

    # ── Data: insurance network (7 binary + 20 categorical concepts) ──
    datamodule = BnLearnDataModule(
        name='insurance',
        seed=42,
        n_gen=10000,
        batch_size=512,
        val_size=0.1,
        test_size=0.2,
    )
    datamodule.setup('fit')

    annotations = datamodule.annotations
    concept_names = annotations.get_axis_annotation(1).labels

    # Assign distribution families to each concept
    axis = annotations.get_axis_annotation(1)
    variable_distributions = {
        name: Bernoulli if axis.cardinalities[i] == 1 else OneHotCategorical
        for i, name in enumerate(concept_names)
    }

    # Shared model kwargs
    model_kwargs = dict(
        input_size=datamodule.dataset.input_data.shape[1],
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['PropCost'],          # no separate task — all nodes are concepts
        latent_encoder_kwargs={'hidden_size': 32, 'n_layers': 2},
        lightning=True,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 1e-3},
    )

    # ── Scenario 1: single loss per type ─────────────────────
    print("=" * 60)
    print("Scenario 1: Single loss per type")
    print("=" * 60)

    loss_fn = ConceptLoss(
        annotations=annotations,
        binary=nn.BCEWithLogitsLoss(),
        categorical=nn.CrossEntropyLoss(),
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)

    # ── Scenario 2: composite losses on both types ───────────
    print("\n" + "=" * 60)
    print("Scenario 2: Per-type composite loss with weights")
    print("=" * 60)

    loss_fn = ConceptLoss(
        annotations=annotations,
        binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        binary_weights=[1.0, 0.5],
        categorical=[nn.CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
        categorical_weights=[1.0, 0.3],
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)

    # ── Scenario 3: regularizer only on binary concepts ──────
    print("\n" + "=" * 60)
    print("Scenario 3: Regularizer only on binary, plain CE on categorical")
    print("=" * 60)

    loss_fn = ConceptLoss(
        annotations=annotations,
        binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.05)],
        binary_weights=[1.0, 0.5],
        categorical=nn.CrossEntropyLoss(),   # single module, no extra weight
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
