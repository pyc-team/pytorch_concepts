"""
Example: Backbone Inside a High-Level Model (frozen or unfrozen)

The same ``Backbone`` used for data-side caching can be passed directly to a
high-level model as its ``backbone``: it runs *inside* the PGM as the
``latent | input`` CPD, consuming raw images end-to-end.

- ``Backbone('resnet18')`` is **frozen** by default: its parameters take no
  gradients and it stays in eval mode even under ``model.train()`` (linear
  probing / feature extraction).
- ``Backbone('resnet18', freeze=False)`` is trainable end-to-end (fine-tuning).
- ``latent_size`` is inferred from ``backbone.out_features`` — no need to
  pass it.

Both regimes are trained with the regular PyTorch Lightning path
(``lightning=True`` + ``Trainer``), and the backbone weights are compared
before/after training to verify that freezing does what it promises.
"""
import torch
from pytorch_lightning import Trainer

from torch_concepts import Backbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.nn import ConceptBottleneckModel


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit(backbone, dm):
    """Build a CBM around ``backbone`` and train it briefly with Lightning."""
    model = ConceptBottleneckModel(
        input_size=dm.n_features,  # (C, H, W) raw images
        annotations=dm.annotations,
        task_names=['Attractive'],
        backbone=backbone,  # latent_size inferred from backbone.out_features
        lightning=True,
        loss=torch.nn.BCEWithLogitsLoss(),
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.01},
    )
    print(f"trainable params: {n_trainable(model):,} "
          f"/ {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=3,      # a few steps are enough to see weights move
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator='cpu',          # torchvision preprocessing is unreliable on MPS
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    return model


def main():
    seed_everything(42)

    # 1. Data: CelebA images
    dm = CelebADataModule(
        root='./data/celeba',
        max_samples=500,
        batch_size=64,
        # splitter=None replaces the native split (defined on the full dataset)
        # with a random split.
        splitter=None,
        seed=42,
    )

    # ── Frozen backbone (default): linear probing ────────────────────────
    frozen_backbone = Backbone('resnet18')
    weight_before = frozen_backbone._model[0].weight.clone()

    model = fit(frozen_backbone, dm)

    # Lightning trained the model, yet the frozen backbone never moved and
    # stays in eval mode even when the model is put in train mode.
    assert torch.equal(weight_before, frozen_backbone._model[0].weight), \
        "frozen backbone weights must not change"
    model.train()
    assert not frozen_backbone.training, \
        "frozen backbone must stay in eval under model.train()"
    print("[frozen]   backbone weights bit-identical after training; stays in eval ✓")

    # ── Unfrozen backbone: end-to-end fine-tuning ────────────────────────
    unfrozen_backbone = Backbone('resnet18', freeze=False)
    weight_before = unfrozen_backbone._model[0].weight.clone()

    model = fit(unfrozen_backbone, dm)

    assert not torch.equal(weight_before, unfrozen_backbone._model[0].weight), \
        "unfrozen backbone weights must update"
    model.train()
    assert unfrozen_backbone.training, "unfrozen backbone must enter train mode"
    print("[unfrozen] backbone weights updated by training; enters train mode ✓")


if __name__ == '__main__':
    main()
