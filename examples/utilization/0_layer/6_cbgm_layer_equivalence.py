"""
CBGM concept bottleneck layer vs. the authors' implementation (Low-Level Interface)
==================================================================================

PyC's :class:`MixConceptEmbeddingToEmbedding` and
:func:`concept_orthogonality` reimplement the two novel pieces of Concept
Bottleneck Generative Models (Ismail et al., ICLR 2024) on top of the library's
existing CEM mixture. This script checks that reimplementation against the
authors' released code, bit for bit.

The reference is https://github.com/prescient-design/CBGM, which ships the
VAE-**GAN** variant (``models/cb_vaegan.py``, paper Eq. 2) rather than the pure
VAE built by ``ConceptBottleneckGenerativeModel`` (Eq. 1). The generative wrapper
therefore differs by construction; what is common to both — and is what this
script pins down — is the concept bottleneck layer itself:

1. **The mixture.** Reference ``CBM_plus_Dec.forward`` slices each concept's
   context into ``concept_bins[c]`` embeddings and sums them weighted by the
   concept probabilities. PyC routes the same computation through
   ``grouped_concept_exogenous_mixture``, the layer CEM already uses.
2. **The unsupervised context.** Both concatenate one extra, unmixed context
   embedding to the bottleneck.
3. **The orthogonality loss.** Reference
   ``utils/gan_loss.py::OrthogonalProjectionLoss``, applied once per concept and
   summed.

Both reference snippets below are transcribed verbatim from those files (only
the surrounding class and the generator call are dropped, since the layer's
output is what is being compared).

Expected outcome:
- Three ``torch.equal`` assertions pass: the mixed concept contexts, the
  unsupervised context, and the orthogonality loss are identical to the
  reference, with no tolerance.
"""
import torch

from torch_concepts import Annotations, seed_everything
from torch_concepts.nn import MixConceptEmbeddingToEmbedding
from torch_concepts.nn.functional import concept_orthogonality

N_CONCEPTS = 4
CONCEPT_BINS = [2, 3, 2, 5]
EMB_SIZE = 8
BATCH = 16


# --------------------------------------------------------------------------
# Reference implementation, transcribed from prescient-design/CBGM
# --------------------------------------------------------------------------
def reference_bottleneck(contexts, probs, n_concepts, emb_size):
    """The mixing half of ``models/cb_vaegan.py::CBM_plus_Dec.forward``.

    ``contexts`` holds one ``(batch, concept_bins[c] * emb_size)`` tensor per
    concept plus, last, the ``(batch, emb_size)`` unsupervised context — i.e.
    what ``self.concept_context_generators`` produces. ``probs`` holds one
    ``(batch, concept_bins[c])`` tensor per concept, i.e. the output of
    ``F.softmax(self.concept_prob_generators[c](context))``.
    """
    non_concept_latent = None
    all_concept_latent = None
    for c in range(n_concepts + 1):
        context = contexts[c]
        if c < n_concepts:
            prob_gumbel = probs[c]
            for i in range(probs[c].shape[1]):
                temp_concept_latent = context[
                    :, (i * emb_size) : ((i + 1) * emb_size)
                ] * prob_gumbel[:, i].unsqueeze(-1)
                if i == 0:
                    concept_latent = temp_concept_latent
                else:
                    concept_latent = concept_latent + temp_concept_latent

            if all_concept_latent == None:
                all_concept_latent = concept_latent
            else:
                all_concept_latent = torch.cat(
                    (all_concept_latent, concept_latent), 1
                )
        else:
            if non_concept_latent == None:
                non_concept_latent = context
            else:
                non_concept_latent = torch.cat((non_concept_latent, context), 1)

    return all_concept_latent, non_concept_latent


def reference_orthogonality(embed1, embed2):
    """``utils/gan_loss.py::OrthogonalProjectionLoss``, verbatim."""
    from torch import nn
    import torch.nn.functional as F

    #  features are normalized
    embed1 = F.normalize(embed1, dim=1)
    embed2 = F.normalize(embed2, dim=1)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = torch.abs(cos(embed1, embed2))
    return output.mean()


# --------------------------------------------------------------------------


def main():
    seed_everything(42)

    # One context block per concept (bins x emb), plus the unsupervised one.
    contexts = [torch.randn(BATCH, bins * EMB_SIZE) for bins in CONCEPT_BINS]
    contexts.append(torch.randn(BATCH, EMB_SIZE))
    probs = [torch.softmax(torch.randn(BATCH, bins), dim=-1) for bins in CONCEPT_BINS]

    print("=" * 70)
    print("Step 1: reference bottleneck (prescient-design/CBGM)")
    print("=" * 70)
    ref_concept_latent, ref_unknown = reference_bottleneck(
        contexts, probs, N_CONCEPTS, EMB_SIZE
    )
    print(f"concept contexts {tuple(ref_concept_latent.shape)}, "
          f"unsupervised context {tuple(ref_unknown.shape)}")

    print("\n" + "=" * 70)
    print("Step 2: PyC MixConceptEmbeddingToEmbedding on the same inputs")
    print("=" * 70)
    annotations = Annotations(
        labels=[f"c{i}" for i in range(N_CONCEPTS)],
        cardinalities=CONCEPT_BINS,
        types=["categorical"] * N_CONCEPTS,
    )
    layer = MixConceptEmbeddingToEmbedding(
        in_concepts=annotations, in_embeddings=EMB_SIZE
    )
    # PyC keeps the states on their own axis: (batch, sum(bins), emb_size).
    # The layer mixes; the unsupervised context is concatenated by the caller
    # (in ``ConceptBottleneckGenerativeModel`` that happens in the decoder CPD).
    mixed = layer(
        concepts=torch.cat(probs, dim=-1),
        embeddings=torch.cat(
            [c.unflatten(-1, (-1, EMB_SIZE)) for c in contexts[:-1]], dim=-2
        ),
    )
    print(f"mixed contexts {tuple(mixed.shape)} = (batch, n_concepts, emb_size)")

    flat = torch.cat([mixed.flatten(start_dim=-2), contexts[-1]], dim=-1)
    print(f"bottleneck {tuple(flat.shape)} = (batch, (n_concepts + 1) * emb_size)")
    assert torch.equal(flat[:, : N_CONCEPTS * EMB_SIZE], ref_concept_latent), \
        "mixed concept contexts differ from the reference"
    print("OK  mixed concept contexts are bit-identical")
    assert torch.equal(flat[:, N_CONCEPTS * EMB_SIZE:], ref_unknown), \
        "unsupervised context differs from the reference"
    print("OK  unsupervised context is bit-identical")

    print("\n" + "=" * 70)
    print("Step 3: orthogonality loss")
    print("=" * 70)
    reference = sum(
        reference_orthogonality(
            ref_concept_latent[:, c * EMB_SIZE: (c + 1) * EMB_SIZE], ref_unknown
        )
        for c in range(N_CONCEPTS)
    )
    pyc = concept_orthogonality(flat, n_concepts=N_CONCEPTS)
    print(f"reference {reference.item():.10f} | pyc {pyc.item():.10f}")
    assert torch.equal(reference, pyc), "orthogonality loss differs from the reference"
    print("OK  orthogonality loss is bit-identical")


if __name__ == "__main__":
    main()
