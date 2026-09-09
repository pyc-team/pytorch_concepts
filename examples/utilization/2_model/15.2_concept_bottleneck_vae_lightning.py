"""
Concept Bottleneck Generative Model (CB-VAE) on Color-MNIST.

Where a CBM/CEM runs ``input -> latent -> concepts -> tasks``, a concept
bottleneck *generative* model runs the other way: ``Z -> C -> X``. A latent
``z ~ N(0, I)`` produces per-concept context embeddings, each concept is
decoded from its own embeddings, the embeddings are mixed by the predicted
concept probabilities (exactly the CEM mixture), and the resulting bottleneck
``w = [w_1, ..., w_k, w_unk]`` is decoded back into the image. The extra
``w_unk`` is the *unsupervised* context: the capacity the pre-defined concepts
do not cover.

Inference is variational: a Pyro guide ``q(z | x)`` supplies the encoder
direction, so the model trains as a VAE with two extra terms — a concept loss
on the bottleneck's concept probabilities and an orthogonality loss pushing the
unsupervised context away from the concept contexts.

References:
    Ismail et al. "Concept Bottleneck Generative Models", ICLR 2024.
    https://openreview.net/forum?id=L9U5MJJleF
"""
import torch
from torch import nn
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.nn import (
    VariationalInference,
    AncestralSamplingInference,
    CompositeLoss,
    ConceptBottleneckVAE,
    ConceptLoss,
    KLDivergenceLoss,
    OrthogonalityLoss,
    MSEReconstructionLoss,
)

# data hparams
MAX_SAMPLES = 30000   # half of MNIST's train split, to keep the demo quick
BATCH_SIZE = 256      # the CBM/CEM range for MNIST-sized images

# model hparams
LATENT_SIZE = 32      # dim(z); MNIST VAEs sit in the 20-64 range
EMBEDDING_SIZE = 16   # dim of one concept embedding; CEM's default m=16
CHANNELS = 32         # base channels, DCGAN-style: doubled per stride-2 stage
USE_UNKNOWN = True    # the paper's w_{k+1}; without it ORT_WEIGHT does nothing

# loss hparams
RECON_WEIGHT = 0.5      # a sigma=1 Gaussian NLL is half the squared error
KL_WEIGHT = 1.0         # plain ELBO: beta = 1
FREE_BITS = 0.5         # KL free bits (nats) per latent dimension
CONC_WEIGHT = 5.0       # concept supervision, within the CBM lambda sweep
ORT_WEIGHT = 1.0        # orthogonality loss weight

# inference hparams
P_INT_TRAIN = 1        # CEM's RandInt rate
TEMPERATURE = 1.0        # initial Gumbel-Softmax temperature
TEMPERATURE_FINAL = 0.5  # its floor, as in Jang et al.
ANNEALING_RATE = 5e-5    # exponential decay per step, reaching the floor late

# training hparams
N_EPOCHS = 100
LEARNING_RATE = 1e-3      # Adam's usual VAE setting
CLIP_GRAD_MAX_NORM = 1.0  # max norm for gradient clipping


def save_grid(top, bottom, path):
    """Write two rows of images, if matplotlib is around."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the grid.")
        return
    # The original keeps its (B, 3, 28, 28) event shape; a generated or
    # reconstructed one comes back flat on the annotated axis, so reshape each.
    images = torch.cat([top.reshape(-1, 3, 28, 28),
                        bottom.reshape(-1, 3, 28, 28)])
    _, axes = plt.subplots(2, len(top), figsize=(len(top), 2))
    for ax, image in zip(axes.flatten(), images):
        ax.imshow(image.permute(1, 2, 0).detach().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"reconstruction grid saved to {path}")


def save_image(image, path):
    """Write a single generated image, if matplotlib is around."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the generated image.")
        return
    plt.figure(figsize=(1.5, 1.5))
    # A Delta observation is unbounded, so clamp before imshow.
    plt.imshow(image.reshape(3, 28, 28).clamp(0, 1).permute(1, 2, 0).detach().numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"generated image saved to {path}")


def main():
    seed_everything(42)

    # `parity` is left out: it is a deterministic function of `digit`, so it adds
    # a redundant bottleneck slot without anything new to steer.
    datamodule = ColorMNISTDataModule(
        root="./data/mnist",
        concept_subset=["color", "digit"],
        max_samples=MAX_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=42,
    )
    datamodule.setup()
    dataset = datamodule.dataset
    concept_names = dataset.concept_names

    context_size = (len(concept_names) + 1) * EMBEDDING_SIZE if USE_UNKNOWN else len(concept_names) * EMBEDDING_SIZE

    model = ConceptBottleneckVAE(
        input_size=dataset.n_features,
        annotations=dataset.annotations,
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        encoder=nn.Sequential(
            nn.Conv2d(3, CHANNELS, 4, stride=2, padding=1), nn.LeakyReLU(),       # 14x14
            nn.Conv2d(CHANNELS, 2 * CHANNELS, 4, stride=2, padding=1), nn.LeakyReLU(),  # 7x7
            nn.Flatten(),
            nn.Linear(2 * CHANNELS * 7 * 7, LATENT_SIZE),
        ),
        decoder=nn.Sequential(
            nn.Linear(context_size, 2 * CHANNELS * 7 * 7), nn.LeakyReLU(),
            nn.Unflatten(1, (2 * CHANNELS, 7, 7)),
            nn.ConvTranspose2d(2 * CHANNELS, CHANNELS, 4, stride=2, padding=1), nn.LeakyReLU(),  # 14x14
            nn.ConvTranspose2d(CHANNELS, 3, 4, stride=2, padding=1),          # 28x28
        ),
        use_unknown=USE_UNKNOWN,
        inference=VariationalInference,
        inference_kwargs={"p_int": 0.0},
        train_inference=VariationalInference,
        train_inference_kwargs={
            "p_int": P_INT_TRAIN,
            "initial_temperature": TEMPERATURE,
            "annealing": "exponential",
            "annealing_rate": ANNEALING_RATE,
            "final_temperature": TEMPERATURE_FINAL,
        },
        lightning=True,
        # --- Lightning-specific arguments ---
        loss=CompositeLoss(
            terms=[
                MSEReconstructionLoss(variable="input"),
                KLDivergenceLoss(latents=["z"], free_bits=FREE_BITS),
                ConceptLoss(categorical=nn.CrossEntropyLoss()),
                OrthogonalityLoss("mixing", "unknown", len(concept_names)) if USE_UNKNOWN else None
            ],
            weights=[RECON_WEIGHT, KL_WEIGHT, CONC_WEIGHT, ORT_WEIGHT],
        ),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": LEARNING_RATE},
    )
    print(model)

    trainer = Trainer(
        max_epochs=N_EPOCHS,
        gradient_clip_val=CLIP_GRAD_MAX_NORM,
        accelerator="mps",
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)

    # Reconstruct a held-out batch: q(z|x) -> concepts -> context -> x.
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    x = batch["inputs"]["x"][:8]
    c = batch["concepts"]["c"][:8]
    with torch.no_grad():
        out = model(query=list(model.pgm.variables), input=x)
    save_grid(x, out.value["input"].clamp(0, 1), "cbvae_colormnist_reconstruction.png")

    # Generate from concepts alone: every digit in both colours. One shared z,
    # so the only thing varying across the grid is the concept intervention.
    model.setup_inference(AncestralSamplingInference)
    with torch.no_grad():
        z = model(query=['z'], evidence={}).samples['z'].tensor
        out = model(
            query=['input'],
            evidence={'z': z.expand(20, -1),
                      'color': torch.eye(2).repeat_interleave(10, 0),  # red row, green row
                      'digit': torch.eye(10).repeat(2, 1)}             # 0..9, 0..9
        )
    red, green = out.value["input"].chunk(2)
    save_grid(red, green, "cbvae_colormnist_conditional_grid.png")

    # Generate a random sample.
    with torch.no_grad():
        out = model(query=['input'], evidence={})
    save_image(out.value["input"], "cbvae_colormnist_random.png")

if __name__ == "__main__":
    main()
