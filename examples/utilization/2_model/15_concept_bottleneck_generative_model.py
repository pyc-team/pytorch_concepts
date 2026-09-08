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

Experiment settings:
- Dataset: Color-MNIST via ``ColorMNISTDataModule``, two categorical concepts
  (``digit`` 10-way, ``color`` 2-way), images flattened to 3x28x28 = 2352
  pixels, decoded deterministically (a ``Delta`` observation scored by
  squared error).
- Model: ``ConceptBottleneckVAE`` with MLP encoder/decoder.
- Inference engine: Pyro ``VariationalInference`` with a guide on ``z``.
- Loss: a ``CompositeLoss`` of ``recon + kl + alpha * concept + beta * orthogonality``.

Expected outcome:
- The ELBO terms go down and both concept accuracies climb well above chance
  (0.1 for ``digit``, 0.5 for ``color``): about 0.94 and 1.00 after 30 epochs.
- The reconstruction grid written at the end shows recognisable coloured
  digits.

References:
    Ismail et al. "Concept Bottleneck Generative Models", ICLR 2024.
    https://openreview.net/forum?id=L9U5MJJleF
"""
import math

import torch

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.nn import (
    AncestralSamplingInference,
    CompositeLoss,
    ConceptBottleneckVAE,
    ConceptLoss,
    KLDivergenceLoss,
    MLP,
    NLLProbLoss,
    OrthogonalityLoss,
    MSEReconstructionLoss,
)

# data hparams
MAX_SAMPLES = 30000
BATCH_SIZE = 1024

# model hparams
LATENT_SIZE = 32      # dim(z)
EMBEDDING_SIZE = 8    # dim of one concept embedding
ENCODER_HIDDEN = 256  # hidden size of the encoder MLP
ENCODER_LAYERS = 2    # number of layers in the encoder MLP
DECODER_HIDDEN = 256  # hidden size of the decoder MLP
DECODER_LAYERS = 2    # number of layers in the decoder MLP
USE_UNKNOWN = True    # whether to include the residual

# loss hparams
RECON_WEIGHT = 1.0      # reconstruction loss weight
KL_WEIGHT = 1.0         # KL loss weight
FREE_BITS = 0.5         # KL free bits (nats) per latent dimension
CONC_WEIGHT = 10.0      # concept loss weight
ORT_WEIGHT = 1.0       # orthogonality loss weight

# training hparams
N_EPOCHS = 100


def save_grid(original, reconstruction, path):
    """Write an originals-over-reconstructions grid, if matplotlib is around."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the reconstruction grid.")
        return
    images = torch.cat([original, reconstruction]).reshape(-1, 3, 28, 28)
    _, axes = plt.subplots(2, len(original), figsize=(len(original), 2))
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
        concept_subset=["digit", "color"],
        max_samples=MAX_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=42,
    )
    datamodule.setup()
    dataset = datamodule.dataset
    concept_names = dataset.concept_names

    n_pixels = math.prod(dataset.n_features)
    context_size = (len(concept_names) + 1) * EMBEDDING_SIZE if USE_UNKNOWN else len(concept_names) * EMBEDDING_SIZE

    model = ConceptBottleneckVAE(
        input_size=n_pixels,
        annotations=dataset.annotations,
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        encoder=MLP(n_pixels, ENCODER_HIDDEN, LATENT_SIZE, n_layers=ENCODER_LAYERS),
        decoder=MLP(context_size, DECODER_HIDDEN, n_pixels, n_layers=DECODER_LAYERS),
        use_unknown=USE_UNKNOWN
    )
    print(model)

    loss_fn = CompositeLoss(
        terms=[
            MSEReconstructionLoss(variable="input"),
            KLDivergenceLoss(latents=["z"], free_bits=FREE_BITS),
            ConceptLoss(categorical=NLLProbLoss()),
            OrthogonalityLoss("mixing", "unknown", len(concept_names)) if USE_UNKNOWN else None
        ],
        weights=[RECON_WEIGHT, KL_WEIGHT, CONC_WEIGHT, ORT_WEIGHT] if USE_UNKNOWN 
        else [RECON_WEIGHT, KL_WEIGHT, CONC_WEIGHT]
    )

    # Every PGM variable is queried: the observed image arrives as `input`
    # (evidence), everything else is latent and reported by the engine.
    var_list = list(model.pgm.variables)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = datamodule.train_dataloader()

    for epoch in range(N_EPOCHS):
        totals = torch.zeros(len(loss_fn.terms))
        for batch in loader:
            x = batch["inputs"]["x"].flatten(1)
            c = batch["concepts"]["c"]

            # p(var_list | x)
            # equivalent to inference.query(query=var_list, evidence={"input": x})
            out = model(query=var_list, input=x)

            # 'breakdown' returns a dict of the individual loss terms
            terms = loss_fn.breakdown(out, c)
            loss = sum(terms.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += torch.tensor([t.item() for t in terms.values()])

        print(f"epoch {epoch:03d} | " + " | ".join(
            f"{name} {value / len(loader):.4f}"
            for name, value in zip(loss_fn.term_names, totals)))

    # Reconstruct a held-out batch: q(z|x) -> concepts -> context -> x.
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    x = batch["inputs"]["x"].flatten(1)[:8]
    c = batch["concepts"]["c"][:8]
    with torch.no_grad():
        out = model(query=var_list, input=x)
    save_grid(x, out.value["input"].clamp(0, 1), "cbvae_colormnist_reconstruction.png")

    # Generate from concepts alone.
    # generate a green 7.
    generator = AncestralSamplingInference(model.pgm)
    with torch.no_grad():
        sampled_z = generator.query(
            query=['z'], 
            evidence={}
        ).samples['z']
        out = generator.query(
            query=['input'], 
            evidence={'z': sampled_z.tensor, 
                      'digit': torch.tensor([[0,0,0,0,0,0,0,1,0,0]]), # 7 
                      'color': torch.tensor([[0,1]])} # green
        )
    save_image(out.value["input"], "cbvae_colormnist_green_seven.png")


if __name__ == "__main__":
    main()
