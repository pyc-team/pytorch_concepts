"""
A Pretrained Backbone
=====================

An ``ImageBackbone`` (a pretrained torchvision or Hugging Face vision model)
can sit inside a high-level model, which then reads raw images end to end; its
output size sets ``latent_size``. By default the backbone is frozen: its
weights get no gradient and it stays in evaluation mode, so only the concept
and task layers learn. With ``freeze=False`` it is fine-tuned with the rest of
the model.

When the backbone stays frozen, it is cheaper to compute the embeddings once
and train on those: see ``data/04_celeba_embeddings.py``.

Data: 500 CelebA images (the dataset, ~1.4 GB, is downloaded on first run),
with the 40 face attributes as concepts and ``Attractive`` as the task. Each
model is trained for a few steps only, enough to see which weights move.
"""

# %%
import torch
from pytorch_lightning import Trainer

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss

seed_everything(42)

# %%
# Data
# ----
datamodule = CelebADataModule(
    root=str(DATA_ROOT / "celeba"),
    max_samples=500,
    batch_size=64,
    seed=42,
    splitter=None,  # required with max_samples
)

# %%
# Frozen and fine-tuned backbones
# -------------------------------
for freeze in (True, False):
    backbone = ImageBackbone("resnet18", freeze=freeze)
    model = ConceptBottleneckModel(
        input_size=datamodule.n_features,  # (3, H, W) images
        annotations=datamodule.annotations,
        task_names=["Attractive"],
        backbone=backbone,
        lightning=True,
        loss=ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    first_weight = next(backbone.parameters()).detach().cpu().clone()

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=datamodule)

    moved = not torch.equal(first_weight, next(backbone.parameters()).detach().cpu())
    model.train()  # a frozen backbone stays in evaluation mode
    print(f"freeze={freeze}")
    print(f"  trainable parameters:      {trainable:,}")
    print(f"  backbone weights changed:  {moved}")
    print(f"  backbone in training mode: {backbone.training}")
