"""
Concept Bottleneck Model
========================

A Concept Bottleneck Model (CBM) first predicts human-interpretable concepts
from the input, then predicts the task from those concepts alone. Here it is
assembled from PyC layers inside a plain ``torch.nn.Module`` and trained with
a plain PyTorch loop.

The data is the XOR toy dataset: two binary concepts ``C1`` and ``C2`` read
off a 2D input, and the task ``xor = C1 XOR C2``. The CBM learns the concepts,
but not the task: its task head is linear, and XOR is not a linear function of
the concepts, so the best the head can do is answer 0.5 everywhere.
``04_concept_embedding_model.py`` lifts this limit.

Reference: Koh et al., "Concept Bottleneck Models", ICML 2020.
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import LinearConceptToConcept, LinearEmbeddingToConcept

seed_everything(42)

# %%
# Data
# ----
dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
x = dataset.input_data  # (1000, 2) inputs
c = dataset.concepts[["C1", "C2"]]  # (1000, 2) concept labels
y = dataset.concepts[["xor"]]  # (1000, 1) task labels

x_train, x_test = x[:800], x[800:]
c_train, c_test = c[:800], c[800:]
y_train, y_test = y[:800], y[800:]


# %%
# Model
# -----
# PyC layers are ``torch.nn`` modules named ``<Operation><Input>To<Output>``.
# Giving them annotations instead of sizes lets them refer to concepts by name.
class ConceptBottleneckModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
        self.concept_encoder = LinearEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=c.annotations,
        )
        self.task_predictor = LinearConceptToConcept(
            in_concepts=c.annotations,
            out_concepts=y.annotations,
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        c_logits = self.concept_encoder(embeddings)
        c_probs = torch.sigmoid(c_logits)
        y_logits = self.task_predictor(c_probs)
        return c_logits, y_logits


model = ConceptBottleneckModel()

# %%
# Training
# --------
# Concepts and task are supervised jointly.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(500):
    optimizer.zero_grad()
    c_logits, y_logits = model(x_train)
    loss = loss_fn(c_logits, c_train) + 0.5 * loss_fn(y_logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
# The task head sees only the concept probabilities, so it can be queried
# directly on the four possible combinations of concepts.
with torch.no_grad():
    c_logits, _ = model(x_test)
    combinations = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    p_xor = torch.sigmoid(model.task_predictor(concepts=combinations))
print(f"test concept accuracy: {((c_logits > 0).float() == c_test).float().mean():.2f}")
for (c1, c2), p in zip(combinations.int().tolist(), p_xor.flatten().tolist()):
    print(f"C1={c1}, C2={c2} -> P(xor) = {p:.2f}  (true: {c1 ^ c2})")
