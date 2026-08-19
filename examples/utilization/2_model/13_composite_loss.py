"""
Example: Composing loss terms, at every level PyC offers

Uses the **insurance** Bayesian-network dataset which has **both** binary
and categorical concepts, making it a good test-bed for type-specific loss
composition.

Three levels of composition, each strictly more general than the last:
  1. Within one type — ``ConceptLoss(binary=[...])`` sums several terms on the
     same type's slice (scenarios 1-3).
  2. Across independent terms — ``CompositeLoss`` sums whole-output terms that
     have nothing to do with per-type routing (scenario 4).
  3. By concept name — ``ConceptSubset`` restricts a loss to a named group,
     so terms can be weighted per *group* rather than per *type*; this is
     exactly what ``WeightedConceptLoss`` builds for you (scenario 5).
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.nn import (
    CompositeLoss,
    ConceptBottleneckModel,
    ConceptLoss,
    ConceptSubset,
    L1LogitRegularizer,
    MLP,
    PyCLoss,
)
from torch_concepts.data import BnLearnDataModule


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
    concept_names = annotations.labels

    # Assign distribution families to each concept
    axis = annotations
    variable_distributions = {
        name: Bernoulli if axis.cardinalities[i] == 1 else OneHotCategorical
        for i, name in enumerate(concept_names)
    }

    # Shared model kwargs
    input_size = datamodule.dataset.input_data.shape[1]
    model_kwargs = dict(
        input_size=input_size,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['PropCost'],          # no separate task — all nodes are concepts
        backbone=MLP(input_size=input_size, hidden_size=32, n_layers=2),
        latent_size=32,
        lightning=True,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 1e-3},
    )

    # ── Scenario 1: single loss per type ─────────────────────
    print("=" * 60)
    print("Scenario 1: Single loss per type")
    print("=" * 60)

    loss_fn = ConceptLoss(
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
        binary=[nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.05)],
        binary_weights=[1.0, 0.5],
        categorical=nn.CrossEntropyLoss(),   # single module, no extra weight
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)

    # ── Scenario 4: CompositeLoss — independent whole-output terms ──
    # A per-type list only composes terms that read *one type's* slice. A term
    # that reads the whole output instead — a shared penalty, an ELBO term —
    # is a separate PyCLoss and belongs in a CompositeLoss, not in a per-type
    # list. Below, concept supervision is one such term; a plain L1 on every
    # reported logit (both types at once, not scoped to either) is another.
    print("\n" + "=" * 60)
    print("Scenario 4: CompositeLoss combining independent terms")
    print("=" * 60)

    class GlobalLogitL1(PyCLoss):
        """L1 over *every* reported logit at once — unlike
        L1LogitRegularizer, which ConceptLoss scopes to one type's slice."""

        def __init__(self, scale: float = 0.01):
            super().__init__()
            self.scale = scale

        def forward(self, output, target=None):
            return self.scale * output.logits.tensor.abs().mean()

    loss_fn = CompositeLoss(
        terms=[
            ConceptLoss(binary=nn.BCEWithLogitsLoss(), categorical=nn.CrossEntropyLoss()),
            GlobalLogitL1(scale=0.01),
        ],
        weights=[1.0, 1.0],
        names=['supervision', 'global_l1'],
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)

    # ── Scenario 5: ConceptSubset — routing by name, not type ────────
    # ConceptLoss only ever routes by *type* (binary/categorical/continuous).
    # Weighting concepts differently from tasks needs the concepts picked out
    # by *name* instead — that's ConceptSubset. Two subsets summed in a
    # CompositeLoss is exactly what WeightedConceptLoss builds internally;
    # written out here to show the general mechanism it is built from.
    print("\n" + "=" * 60)
    print("Scenario 5: ConceptSubset — concepts vs. task, weighted separately")
    print("=" * 60)

    loss_fn = CompositeLoss(
        terms=[
            ConceptSubset(
                ConceptLoss(binary=nn.BCEWithLogitsLoss(), categorical=nn.CrossEntropyLoss()),
                exclude=['PropCost'],
            ),
            ConceptSubset(
                ConceptLoss(binary=nn.BCEWithLogitsLoss(), categorical=nn.CrossEntropyLoss()),
                names=['PropCost'],
            ),
        ],
        weights=[0.5, 1.0],
        names=['concepts', 'task'],
    )
    print(loss_fn)

    model = ConceptBottleneckModel(**model_kwargs, loss=loss_fn)
    Trainer(max_epochs=20, enable_progress_bar=True).fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
