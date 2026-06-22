"""
Example: Verifying automatic routing of model outputs to loss terms

Creates a **fully synthetic** setup with 2 binary + 2 categorical concepts
and a model that returns extra keys (``embeddings``, ``latent``) via
``ModelOutput.extra`` alongside the standard ``logits`` / ``target``.

Four custom loss terms are defined, each declaring a **different subset**
of kwargs in its ``forward()`` signature:

  - ``PlainBCE``        — accepts only (input, target)
  - ``EmbeddingReg``    — accepts (input, embeddings)   → L2 on embeddings
  - ``LatentReg``       — accepts (input, latent)        → L1 on latent
  - ``KitchenSinkReg``  — accepts **kwargs               → sees everything

The signature-based dispatch in ``ConceptLoss._compute_type_loss`` should
route each term only the kwargs it declares, so:

  - ``PlainBCE`` never sees ``embeddings`` or ``latent``
  - ``EmbeddingReg`` never sees ``target`` or ``latent``
  - ``LatentReg`` never sees ``target`` or ``embeddings``
  - ``KitchenSinkReg`` sees all keys
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, MLP


# ======================================================================
# Custom loss terms with different forward() signatures
# ======================================================================

class PlainBCE(nn.Module):
    """Standard BCE — only needs (input, target)."""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(input, target)


class EmbeddingReg(nn.Module):
    """L2 penalty on ``embeddings`` — does NOT accept target."""
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        print(f"    [EmbeddingReg] received keys: input{list(input.shape)}, "
              f"embeddings{list(embeddings.shape)}")
        return self.scale * embeddings.pow(2).mean()


class LatentReg(nn.Module):
    """L1 penalty on ``latent`` — does NOT accept target or embeddings."""
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        print(f"    [LatentReg]    received keys: input{list(input.shape)}, "
              f"latent{list(latent.shape)}")
        return self.scale * latent.abs().mean()


class KitchenSinkReg(nn.Module):
    """Accepts **kwargs — sees everything the loss dispatches."""
    def __init__(self, scale: float = 0.001):
        super().__init__()
        self.scale = scale

    def forward(self, **kwargs) -> torch.Tensor:
        print(f"    [KitchenSink]  received keys: {sorted(kwargs.keys())}")
        return self.scale * kwargs['input'].abs().mean()


# ======================================================================
# Model subclass that exposes extra outputs for loss
# ======================================================================

class CBMWithExtraOutputs(ConceptBottleneckModel):
    """Adds ``embeddings`` and ``latent`` to the model output extras."""

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        batch_size = out.logits.shape[0]
        device = out.logits.device
        # Synthetic extra outputs (in practice these come from the model)
        out.extra = {
            'embeddings': torch.randn(batch_size, 16, device=device),
            'latent': torch.randn(batch_size, 8, device=device),
        }
        return out


# ======================================================================
# Synthetic data
# ======================================================================

def make_synthetic_data(n=512, seed=0):
    """2 binary + 2 categorical concepts (cardinalities 1,1,3,4) → 9 logits."""
    torch.manual_seed(seed)
    x = torch.randn(n, 10)
    c_binary = torch.randint(0, 2, (n, 2)).float()
    c_cat1 = torch.randint(0, 3, (n, 1)).float()
    c_cat2 = torch.randint(0, 4, (n, 1)).float()
    c = torch.cat([c_binary, c_cat1, c_cat2], dim=1)  # (n, 4)
    return x, c


def make_annotations():
    return Annotations({1: AxisAnnotation(
        labels=['b1', 'b2', 'cat1', 'cat2'],
        cardinalities=[1, 1, 3, 4],
        types=['binary', 'binary', 'categorical', 'categorical'],
    )})


# ======================================================================
# Main
# ======================================================================

def main():
    x, c = make_synthetic_data()
    ann = make_annotations()

    # ── Build a composite loss ────────────────────────────────
    # Binary:      PlainBCE + EmbeddingReg + KitchenSinkReg
    # Categorical: CrossEntropyLoss + LatentReg
    loss_fn = ConceptLoss(
        annotations=ann,
        binary=[
            PlainBCE(), 
            EmbeddingReg(scale=0.01), 
            KitchenSinkReg(scale=0.001)
        ],
        binary_weights=[1.0, 0.5, 0.1],
        categorical=[
            nn.CrossEntropyLoss(), 
            LatentReg(scale=0.01)
        ],
        categorical_weights=[1.0, 0.3],
    )
    print("Loss function:", loss_fn)
    print()

    # ── Build model ───────────────────────────────────────────
    # Distributions are owned by the model: binary -> Bernoulli, categorical ->
    # OneHotCategorical are resolved from the concept types (no need to specify).
    model = CBMWithExtraOutputs(
        input_size=10,
        annotations=ann,
        task_names=[],
        backbone=MLP(input_size=10, hidden_size=16, n_layers=1),
        latent_size=16,
    )

    # ── Manual forward + loss (no Lightning, just to inspect routing) ─
    model.train()
    out = model(input=x, query=['b1', 'b2', 'cat1', 'cat2'])
    out.target = c

    print(f"ModelOutput fields: logits={out.logits.shape}, target={out.target.shape}")
    print(f"  extra keys: {sorted(out.extra.keys())}")
    print(f"  embeddings shape: {out.extra['embeddings'].shape}")
    print(f"  latent shape: {out.extra['latent'].shape}")
    print()

    print("Computing loss (watch which keys each term receives):")
    print("-" * 60)
    loss = loss_fn(out)
    print("-" * 60)
    print(f"\nTotal loss: {loss.item():.4f}")

    # Verify gradients flow
    loss.backward()
    print("Backward pass succeeded — gradients flow through all terms.")


if __name__ == "__main__":
    main()
