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
  pixels modelled as a Gaussian observation.
- Model: ``ConceptBottleneckGenerativeModel`` with MLP encoder/decoder.
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
from torch.distributions import Normal

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.nn import (
    CompositeLoss,
    ConceptBottleneckGenerativeModel,
    ConceptLoss,
    KLDivergenceLoss,
    MLP,
    NLLProbLoss,
    OrthogonalityLoss,
    ReconstructionLoss,
)

LATENT_SIZE = 32     # dim(z)
EMBEDDING_SIZE = 16  # m, the width of one context embedding
ALPHA = 5.0          # concept loss weight
BETA = 1.0           # orthogonality loss weight


def concept_accuracy(output, concepts, names):
    return {
        name: (output.probs[name].argmax(-1) == concepts[:, i]).float().mean().item()
        for i, name in enumerate(names)
    }


def main():
    seed_everything(42)

    # `parity` is left out: it is a deterministic function of `digit`, so it adds
    # a redundant bottleneck slot without anything new to steer.
    datamodule = ColorMNISTDataModule(
        root="./data/mnist",
        concept_subset=["digit", "color"],
        max_samples=30000,
        batch_size=256,
        seed=42,
    )
    datamodule.setup()
    dataset = datamodule.dataset
    concept_names = dataset.concept_names

    # The PGM keeps every variable on a single feature axis, so the image is a
    # flat vector of pixels rather than a (3, 28, 28) tensor.
    n_pixels = math.prod(dataset.n_features)
    context_size = (len(concept_names) + 1) * EMBEDDING_SIZE

    model = ConceptBottleneckGenerativeModel(
        input_size=n_pixels,
        annotations=dataset.annotations,
        encoder=MLP(n_pixels, 256, LATENT_SIZE),
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        # A Gaussian likelihood over pixel intensities: `loc` is the decoder's
        # output and `scale` is one fixed sigma shared by every pixel, which
        # keeps the reconstruction-to-KL ratio a hyper-parameter rather than
        # something a learned sigma quietly anneals away.
        observation=Normal,
        scale_init=0.3,
        scale_learnable=False,
        # Raw: the model composes the observation parameter's activation on top
        # (the identity for a Normal's `loc`), so the MLP must not squash its
        # own output — the network learns to land in [0, 1] itself.
        decoder=MLP(context_size, 256, n_pixels, n_layers=2, activation="leaky_relu"),
    )
    print(model)

    # The ELBO, as four independent terms. This model reports `probs`
    # (`param_for_discrete_var`), which the concept term picks up on its own — it
    # only needs a term that scores probabilities rather than logits.
    loss_fn = CompositeLoss(
        terms=[
            ReconstructionLoss(variable="input"),
            KLDivergenceLoss(latents=["z"]),
            ConceptLoss(categorical=NLLProbLoss()),
            OrthogonalityLoss(variables=["mixing", "unknown"]),
        ],
        weights=[1.0, 1.0, ALPHA, BETA],
    )

    # Every PGM variable is queried: the observed image arrives as `input`
    # (evidence), everything else is latent and reported by the engine.
    query = list(model.pgm.variables)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = datamodule.train_dataloader()

    for epoch in range(30):
        totals = torch.zeros(4)
        for batch in loader:
            x = batch["inputs"]["x"].flatten(1)
            c = batch["concepts"]["c"]
            # ReconstructionLoss scores the observed image; the model's
            # `default_extra` publishes it to `out.extra` on every forward call.
            out = model(query=query, input=x)

            # `breakdown` is `loss_fn(out, c)` with the addends kept apart, so the
            # per-term values can be tracked — an ELBO whose KL has collapsed
            # looks fine in the total.
            terms = loss_fn.breakdown(out, c)
            loss = sum(terms.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += torch.tensor([t.item() for t in terms.values()])

        totals /= len(loader)
        print(f"epoch {epoch:03d} | " + " | ".join(
            f"{name} {value:.4f}" for name, value in zip(loss_fn.term_names, totals)))

    # Reconstruct a held-out batch: q(z|x) -> concepts -> context -> x.
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    x = batch["inputs"]["x"].flatten(1)[:8]
    c = batch["concepts"]["c"][:8]
    with torch.no_grad():
        out = model(query=query, input=x)
    print("concept accuracy:", concept_accuracy(out, c, concept_names))
    save_grid(x, out.loc["input"].clamp(0, 1), "cbgm_colormnist_reconstruction.png")


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


if __name__ == "__main__":
    main()
