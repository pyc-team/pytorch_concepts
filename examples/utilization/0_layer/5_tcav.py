"""
Testing with Concept Activation Vectors (Low-Level Interface)
=============================================================

TCAV (Kim et al., ICML 2018) quantifies how important a human concept is to
a trained classifier -- post hoc, without retraining. A Concept Activation
Vector (CAV) is fit as the unit normal of a linear probe separating
concept-positive from concept-negative activations; the TCAV score of a
concept for a class is the fraction of class examples whose class logit
increases along the CAV (a directional derivative).

This example runs the full TCAV protocol on a CelebA subset:

1. Train a small MLP head on frozen ResNet18 embeddings (the model to test).
2. Fit CAVs for three concepts on the training activations
   (``CAVEmbeddingToConcept``) and check the probes are accurate.
3. Compute TCAV scores of each concept for the task class
   (``tcav_score``).
4. Statistical significance (paper Sec. 3.2): repeat with CAVs fit on
   random labels and t-test the real scores against the random ones --
   concepts whose scores are indistinguishable from random are discarded.

Key Components:
- CAVEmbeddingToConcept: post-hoc concept encoder holding one unit-norm CAV
  per concept (PyC's ``torch_concepts.nn``)
- tcav_score: directional-derivative testing of a downstream head along
  the CAVs (PyC's ``torch_concepts.nn.functional``)

Dataset: CelebA (a 4000-image subset, frozen ResNet18 embeddings)
Concepts: Smiling, Male, Blond_Hair
Task: predict 'Attractive'; TCAV asks which concepts the classifier uses
"""
import torch
from scipy.stats import ttest_ind

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.nn import CAVEmbeddingToConcept
from torch_concepts.nn.functional import tcav_score

CONCEPTS = ["Smiling", "Male", "Blond_Hair"]
TASK = "Attractive"
N_RUNS = 10  # CAV fits per concept for the significance test


def load_embeddings(n_samples=10000):
    """Frozen ResNet18 embeddings + binary attributes for a CelebA subset."""
    dm = CelebADataModule(
        root="./data/celeba",
        concept_subset=CONCEPTS + [TASK],
        max_samples=n_samples,
        splitter=None,   # required alongside max_samples (see CelebADataModule docs)
    )
    dm.precompute_embeddings(ImageBackbone("resnet18"), cache=True)
    return dm.dataset.input_data, dm.dataset.concepts.float()


def train_head(x_train, y_train, epochs=20, batch_size=256):
    """Train a small MLP task head on the frozen embeddings."""
    head = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for batch in torch.randperm(len(x_train)).split(batch_size):
            head.train()
            optimizer.zero_grad()
            loss_fn(head(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
    return head.eval()


def fit_cavs(x, c, seed):
    """Fit one CAV per concept on a random half of the data.

    Subsampling makes each run's CAVs differ, as the paper's repeated runs
    against different random negative sets do.
    """
    torch.manual_seed(seed)
    subset = torch.randperm(len(x))[: len(x) // 2]
    layer = CAVEmbeddingToConcept(
        in_embeddings=x.shape[1], out_concepts=c.shape[1]
    )
    accuracy = layer.fit(x[subset], c[subset])
    return layer, accuracy


def main():
    seed_everything(7)

    print("Loading CelebA subset + extracting frozen ResNet18 embeddings...")
    embeddings, attrs = load_embeddings()
    n_train = int(0.8 * len(embeddings))
    x_train, x_test = embeddings[:n_train], embeddings[n_train:]
    c_train = attrs[:n_train, :-1]
    y_train, y_test = attrs[:n_train, -1:], attrs[n_train:, -1:]

    print(f"Training the '{TASK}' MLP head on frozen embeddings...")
    head = train_head(x_train, y_train)
    with torch.no_grad():
        acc = ((head(x_test) > 0).float() == y_test).float().mean().item()
    print(f"  task accuracy: {acc:.2f}")

    # TCAV examines the target-class examples: how does the class logit
    # react to a step towards each concept, at those inputs?
    class_embeddings = x_test[y_test[:, 0] == 1]

    # one reference fit on the full training set, to report probe quality
    cav_layer, accuracy = fit_cavs(x_train, c_train, seed=0)
    print("\nCAV probe accuracy (should be high, or the concept is not "
          "linearly represented):")
    for concept, a in zip(CONCEPTS, accuracy):
        print(f"  {concept:<12} {a:.2f}")

    # significance protocol: N_RUNS scores from real CAVs (subsampled fits)
    # vs N_RUNS scores from CAVs fit on permuted (random) labels
    real_scores, random_scores = [], []
    for run in range(N_RUNS):
        layer, _ = fit_cavs(x_train, c_train, seed=run)
        real_scores.append(tcav_score(class_embeddings, head, layer.cavs))

        c_random = c_train[torch.randperm(len(c_train))]  # break the pairing
        layer, _ = fit_cavs(x_train, c_random, seed=run)
        random_scores.append(tcav_score(class_embeddings, head, layer.cavs))
    real_scores = torch.stack(real_scores)      # (N_RUNS, n_concepts)
    random_scores = torch.stack(random_scores)

    # Welch's t-test (unequal variances) with the paper's Bonferroni
    # correction across concepts. Caveat shared with the original protocol:
    # runs overlap in training data and are scored on the same test set, so
    # they are not fully independent and p-values are optimistic.
    alpha = 0.05 / len(CONCEPTS)
    print(f"\nTCAV scores for class '{TASK}' ({N_RUNS} runs, mean +- std; "
          f"0.5 = irrelevant, alpha = {alpha:.4f}):\n"
          f"{'concept':<14}{'TCAV':<16}{'random CAVs':<16}{'p-value':<10}")
    for j, concept in enumerate(CONCEPTS):
        p_value = ttest_ind(real_scores[:, j], random_scores[:, j],
                            equal_var=False).pvalue
        if torch.isnan(torch.tensor(p_value)):
            # both groups constant (e.g. all runs saturate at 1.0):
            # significant iff the constants differ
            p_value = float(real_scores[:, j].mean()
                            == random_scores[:, j].mean())
        significant = "significant" if p_value < alpha else "NOT significant"
        print(
            f"{concept:<14}"
            f"{real_scores[:, j].mean():.2f} +- {real_scores[:, j].std():.2f}   "
            f"{random_scores[:, j].mean():.2f} +- {random_scores[:, j].std():.2f}   "
            f"{p_value:<10.3f}{significant}"
        )

    print(
        "\nScores above 0.5 mean the classifier's logit increases towards "
        "the concept\nfor most class examples; only concepts whose scores "
        "differ significantly from\nthe random-CAV baseline should be "
        "trusted (paper Sec. 3.2)."
    )


if __name__ == "__main__":
    main()
