"""
A Custom Concept Layer
======================

Every PyC layer extends ``BaseConceptLayer``: it takes concepts and/or
embeddings as input and returns concepts. Subclassing it is all it takes to
write a new one. The base class stores the input and output sizes, whether
they are given as integers or as ``Annotations``; with annotations, the layer
also knows the names of its output concepts and can label its outputs.
"""

# %%
import torch

import torch_concepts as pyc
from torch_concepts import seed_everything

seed_everything(42)


# %%
# Defining the layer
# ------------------
# A linear map from the concatenated concepts and embeddings to new concepts.
class LinearConceptEmbeddingToConcept(pyc.nn.BaseConceptLayer):
    def __init__(self, in_concepts, in_embeddings, out_concepts):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.linear = torch.nn.Linear(
            self.in_concepts_shape + self.in_embeddings_shape,
            self.out_concepts_shape,
        )

    def forward(self, concepts, embeddings):
        return self.linear(torch.cat([concepts, embeddings], dim=-1))


concepts = torch.rand(8, 3)
embeddings = torch.randn(8, 5)

# %%
# Sizes or annotations
# --------------------
layer = LinearConceptEmbeddingToConcept(in_concepts=3, in_embeddings=5, out_concepts=2)
print(layer(concepts=concepts, embeddings=embeddings).shape)

# With annotations the layer can name its outputs. Embeddings are never
# annotated: they are always given by their size.
layer = LinearConceptEmbeddingToConcept(
    in_concepts=pyc.Annotations(labels=["c1", "c2", "c3"], cardinalities=[1, 1, 1]),
    in_embeddings=5,
    out_concepts=pyc.Annotations(labels=["y1", "y2"], cardinalities=[1, 1]),
)
output = layer.annotate(layer(concepts=concepts, embeddings=embeddings))
print(output["y2"].shape, output.annotations.labels)
