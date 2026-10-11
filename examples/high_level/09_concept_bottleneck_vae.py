"""
Concept Bottleneck VAE
======================

A concept bottleneck generative model runs the other way round from a CBM:
``z -> concepts -> image``. A latent ``z`` produces an embedding per concept,
each concept is decoded from its embeddings, the embeddings are mixed by the
concept probabilities (as in a CEM), and the image is decoded from the mixed
embeddings and the concepts. An encoder ``q(z | image)`` makes it a VAE,
trained with variational inference on a reconstruction loss, a KL term and a
concept loss.

Because the image is generated *from* the concepts, setting them generates an
image with the chosen digit and color.

Data: Color-MNIST with the concepts ``digit`` and ``color``. MNIST (~60 MB) is
downloaded on first run; the images are saved as PNG files if matplotlib is
installed.

Reference: Ismail et al., "Concept Bottleneck Generative Models", ICLR 2024.
"""

# %%
import torch
from pytorch_lightning import Trainer
from torch import nn

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    AncestralSamplingInference,
    CompositeLoss,
    ConceptBottleneckVAE,
    ConceptLoss,
    KLDivergenceLoss,
    MSEReconstructionLoss,
    VariationalInference,
)

seed_everything(42)


def save_grid(rows, path):
    """Save rows of (n, 3, 28, 28) images as one PNG, if matplotlib is installed."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return print(f"matplotlib not installed: {path} not saved")
    images = torch.cat([row.reshape(-1, 3, 28, 28) for row in rows])
    _, axes = plt.subplots(len(rows), len(rows[0]), figsize=(len(rows[0]), len(rows)))
    for ax, image in zip(axes.flatten(), images):
        ax.imshow(image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


# %%
# Data
# ----
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    concept_subset=["digit", "color"],
    max_samples=20000,
    batch_size=512,
)

# %%
# Model
# -----
# Convolutional encoder and decoder; the decoder reads the 2 mixed concept
# embeddings (2 x 16) and the concepts themselves (10 + 2). ``use_unknown=False``
# drops the extra context for what the concepts do not cover.
model = ConceptBottleneckVAE(
    input_size=datamodule.n_features,
    annotations=datamodule.annotations,
    latent_size=32,
    embedding_size=16,
    encoder=nn.Sequential(
        nn.Conv2d(3, 32, 4, stride=2, padding=1),  # -> 14 x 14
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 7 x 7
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 32),
    ),
    decoder=nn.Sequential(
        nn.Linear(2 * 16 + 10 + 2, 64 * 7 * 7),
        nn.LeakyReLU(),
        nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 14 x 14
        nn.LeakyReLU(),
        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # -> 28 x 28
    ),
    use_unknown=False,
    concepts_to_decoder=True,
    inference=VariationalInference,
    inference_kwargs={"initial_temperature": 0.1},
    train_inference=VariationalInference,
    train_inference_kwargs={"p_int": 1.0},  # decode from the true concepts in training
    lightning=True,
    loss=CompositeLoss(
        terms=[
            MSEReconstructionLoss(variable="input"),
            KLDivergenceLoss(latents=["z"], free_bits=0.05),
            ConceptLoss(categorical=nn.CrossEntropyLoss()),
        ],
        weights=[0.5, 1.0, 5.0],
        names=["reconstruction", "kl", "concepts"],
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 1e-3},
)

# %%
# Training
# --------
trainer = Trainer(max_epochs=30, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)

# %%
# Reconstruction
# --------------
# image -> q(z | image) -> concepts -> image, on held-out images. The image is
# redrawn from its predicted concepts: a misread digit comes back as the digit
# the model read.
model.eval()
test = datamodule.testset.indices
x_test = datamodule.dataset.input_data[test].to(model.device)
c_test = datamodule.dataset.concepts[test]
with torch.no_grad():
    out = model(query=list(model.pgm.variables), input=x_test)
error = ((out.value["input"] - x_test.flatten(1)) ** 2).mean()
print(f"test reconstruction error (MSE per pixel): {error:.4f}")
for name in ["digit", "color"]:
    predicted = out.logits[name].argmax(-1, keepdim=True).cpu()
    print(f"test accuracy of {name}: {(predicted == c_test[name]).float().mean():.3f}")
save_grid([x_test[:10], out.value["input"][:10]], "cbvae_reconstruction.png")

# %%
# Generation from concepts
# ------------------------
# Sample z from the prior and set the concepts: every digit, in red (top row)
# and in green (bottom row), sharing one z per column.
model.setup_inference(AncestralSamplingInference)
with torch.no_grad():
    z = torch.randn(10, 32, device=model.device).repeat(2, 1)
    digit = torch.eye(10, device=model.device).repeat(2, 1)
    color = torch.eye(2, device=model.device).repeat_interleave(10, 0)
    evidence = {"z": z, "digit": digit, "color": color}
    images = model(query=["input"], evidence=evidence).value["input"]
save_grid(list(images.chunk(2)), "cbvae_generation.png")
