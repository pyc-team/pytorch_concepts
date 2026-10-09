"""
Annotations and Annotated Tensors
=================================

Concept-based models work with tensors whose columns are concepts.
``Annotations`` name those columns and record each concept's type (binary,
categorical or continuous) and cardinality (its number of states). An
``AnnotatedTensor`` carries its annotations along, so that columns are selected
by concept name or type instead of by position. Layers, losses, metrics and
models all read concepts this way.
"""

# %%
import torch

import torch_concepts as pyc

# %%
# Annotations
# -----------
# A categorical concept spans one column per state; binary and continuous
# concepts span one column each.
annotations = pyc.Annotations(
    labels=["smoking", "genotype", "tar", "cancer"],
    types=["binary", "categorical", "continuous", "binary"],
    cardinalities=[1, 3, 1, 1],
)
print(annotations.concept("genotype"))
print(annotations.concept_slices)

# %%
# Annotated tensors
# -----------------
# Model outputs have one column per state: 1 + 3 + 1 + 1 = 6 columns.
logits = pyc.AnnotatedTensor(torch.randn(4, 6), annotations)
print("Genotype slice:", logits["genotype"].shape)  # one concept
print("Smoking and cancer slice:", logits["smoking", "cancer"].shape)  # several concepts
print("Binary concepts:", logits.binary().annotations.labels)  # all concepts of a type

# Tensor methods and arithmetic keep the annotations; ``torch.*`` functions and
# modules return plain tensors, so layers never see them.
print("Sigmoid of logits:", type(logits.sigmoid() * 2).__name__, type(torch.sigmoid(logits)).__name__)
# %%
# Labels
# ------
# Ground-truth labels have one column per concept instead, holding the class
# index of a categorical concept: their annotations are ``to_concept_space()``.
labels = pyc.AnnotatedTensor(
    torch.tensor([[1.0, 2.0, 0.7, 0.0]]),
    annotations.to_concept_space(),
)
print(labels["genotype"])

# %%
# Combining tensors
# -----------------
asbestos = pyc.AnnotatedTensor(
    torch.zeros(4, 1),
    pyc.Annotations(labels=["asbestos"], cardinalities=[1]),
)
print(logits.union_with(asbestos).annotations.labels)
