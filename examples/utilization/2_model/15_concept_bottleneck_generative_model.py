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
  Bernoulli pixels.
- Model: ``ConceptBottleneckGenerativeModel`` with MLP encoder/decoder.
- Inference engine: Pyro ``VariationalInference`` with a guide on ``z``.
- Loss: ``recon + kl + alpha * concept + beta * orthogonality``.

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
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.nn import ConceptBottleneckGenerativeModel, MLP
from torch_concepts.nn.functional import concept_orthogonality

LATENT_SIZE = 32     # dim(z)
EMBEDDING_SIZE = 16  # m, the width of one context embedding
ALPHA = 5.0          # concept loss weight
BETA = 1.0           # orthogonality loss weight


def concept_loss(output, concepts, names):
    """Negative log-likelihood of the ground-truth concepts under the bottleneck.

    The model reports concept *probabilities* (as the reference implementation
    does), so this is an explicit NLL rather than ``CrossEntropyLoss``.
    """
    return sum(
        F.nll_loss(output.probs[name].clamp_min(1e-8).log(), concepts[:, i].long())
        for i, name in enumerate(names)
    )


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
    # flat vector of Bernoulli pixels rather than a (3, 28, 28) tensor.
    n_pixels = math.prod(dataset.n_features)
    context_size = (len(concept_names) + 1) * EMBEDDING_SIZE

    model = ConceptBottleneckGenerativeModel(
        input_size=n_pixels,
        annotations=dataset.annotations,
        encoder=MLP(n_pixels, 256, LATENT_SIZE),
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        observation=Bernoulli,
        # Raw: the model composes the observation parameter's activation on top
        # (here a sigmoid, for a Bernoulli's `probs`), so the MLP must not
        # squash its own output or it would be activated twice.
        decoder=MLP(context_size, 256, n_pixels, n_layers=2, activation="leaky_relu"),
    )
    print(model)

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
            out = model(query=query, input=x)

            # Reconstruction and the Gaussian KL of the guide vs the N(0, I) prior.
            recon = F.binary_cross_entropy(
                out.probs["input"], x, reduction="none"
            ).sum(-1).mean()
            kl = kl_divergence(
                Normal(out.guide_params["loc"]["z"], out.guide_params["scale"]["z"]),
                Normal(out.loc["z"], out.scale["z"]),
            ).sum(-1).mean()
            concept = concept_loss(out, c, concept_names)
            # The bottleneck is the mixed concept contexts followed by the
            # unsupervised one; the PGM keeps them as two variables.
            orthogonality = concept_orthogonality(
                torch.cat([out.value["mixing"], out.value["unknown"]], dim=-1),
                len(concept_names),
            )

            loss = recon + kl + ALPHA * concept + BETA * orthogonality
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += torch.tensor([recon.item(), kl.item(), concept.item(),
                                    orthogonality.item()])

        totals /= len(loader)
        print(f"epoch {epoch:03d} | recon {totals[0]:8.2f} | kl {totals[1]:6.2f} "
              f"| concept {totals[2]:.4f} | orth {totals[3]:.4f}")

    # Reconstruct a held-out batch: q(z|x) -> concepts -> context -> x.
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    x = batch["inputs"]["x"].flatten(1)[:8]
    c = batch["concepts"]["c"][:8]
    with torch.no_grad():
        out = model(query=query, input=x)
    print("concept accuracy:", concept_accuracy(out, c, concept_names))
    save_grid(x, out.probs["input"], "cbgm_colormnist_reconstruction.png")


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
