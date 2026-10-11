"""
Testing with Concept Activation Vectors (TCAV)
==============================================

TCAV measures how much a trained classifier relies on a human concept, post
hoc and without retraining. A Concept Activation Vector (CAV) is the normal of
a linear probe separating concept-positive from concept-negative activations;
the TCAV score of a concept for a class is the fraction of class examples whose
class logit increases along the CAV.

The full protocol runs below: train a classifier, fit one CAV per concept
(``CAVEmbeddingToConcept``), score the concepts (``tcav_score``), and keep only
the concepts whose scores differ significantly from those of CAVs fit on
random labels.

Data: 4,000 CelebA images (the dataset, ~1.4 GB, is downloaded on first run),
embedded by a frozen ResNet18. Concepts: Smiling, Male, Blond_Hair. Class:
Attractive.

Reference: Kim et al., "Interpretability Beyond Feature Attribution:
Quantitative Testing with Concept Activation Vectors (TCAV)", ICML 2018.
"""

# %%
import torch
from scipy.stats import ttest_ind

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import CAVEmbeddingToConcept
from torch_concepts.nn.functional import tcav_score

seed_everything(7)

# %%
# Data
# ----
concepts, task = ["Smiling", "Male", "Blond_Hair"], "Attractive"
datamodule = CelebADataModule(
    root=str(DATA_ROOT / "celeba"),
    concept_subset=concepts + [task],
    max_samples=4000,
    seed=7,
    splitter=None,  # required with max_samples
)
datamodule.precompute_embeddings(ImageBackbone("resnet18"), cache=True)

x = datamodule.dataset.input_data
c = datamodule.dataset.concepts[concepts].float()
y = datamodule.dataset.concepts[[task]].float()
n_train = int(0.8 * len(x))
x_train, x_test = x[:n_train], x[n_train:]
c_train = c[:n_train]
y_train, y_test = y[:n_train], y[n_train:]

# %%
# The classifier to test
# ----------------------
classifier = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()
for _ in range(20):
    for batch in torch.randperm(n_train).split(256):
        optimizer.zero_grad()
        loss_fn(classifier(x_train[batch]), y_train[batch]).backward()
        optimizer.step()
classifier.eval()
with torch.no_grad():
    accuracy = ((classifier(x_test) > 0).float() == y_test).float().mean()
print(f"classifier accuracy: {accuracy:.2f}")


# %%
# Concept activation vectors
# --------------------------
# One CAV per concept, fit on a random half of the training set: repeated fits
# differ, as the paper's repeated runs do. A CAV is only meaningful if its
# probe is accurate, i.e. if the concept is linearly represented.
def fit_cavs(labels, seed):
    torch.manual_seed(seed)
    half = torch.randperm(n_train)[: n_train // 2]
    layer = CAVEmbeddingToConcept(in_embeddings=x.shape[1], out_concepts=len(concepts))
    probe_accuracy = layer.fit(x_train[half], labels[half])
    return layer.cavs, probe_accuracy


_, probe_accuracy = fit_cavs(c_train, seed=0)
for concept, value in zip(concepts, probe_accuracy):
    print(f"CAV probe accuracy for {concept}: {float(value):.2f}")

# %%
# TCAV scores and significance
# ----------------------------
# TCAV looks at the class examples: does the class logit grow when they move
# towards a concept? The scores of 10 CAV fits are compared (Welch's t-test,
# Bonferroni-corrected) against 10 fits on randomly permuted concept labels.
class_examples = x_test[y_test[:, 0] == 1]
scores, random_scores = [], []
for run in range(10):
    cavs, _ = fit_cavs(c_train, seed=run)
    scores.append(tcav_score(class_examples, classifier, cavs))
    cavs, _ = fit_cavs(c_train[torch.randperm(n_train)], seed=run)
    random_scores.append(tcav_score(class_examples, classifier, cavs))
scores = torch.stack(scores)  # (runs, concepts)
random_scores = torch.stack(random_scores)

alpha = 0.05 / len(concepts)
print(f"TCAV scores for '{task}' (0.5 = no influence):")
print(f"  {'concept':<11} {'TCAV':<13} {'random CAVs':<13} p-value")
for j, concept in enumerate(concepts):
    p_value = ttest_ind(scores[:, j], random_scores[:, j], equal_var=False).pvalue
    if p_value != p_value:  # NaN: both groups constant, significant iff they differ
        p_value = float(scores[:, j].mean() == random_scores[:, j].mean())
    verdict = "significant" if p_value < alpha else "not significant"
    real = f"{scores[:, j].mean():.2f} +- {scores[:, j].std():.2f}"
    random = f"{random_scores[:, j].mean():.2f} +- {random_scores[:, j].std():.2f}"
    print(f"  {concept:<11} {real:<13} {random:<13} {p_value:.3f} ({verdict})")
