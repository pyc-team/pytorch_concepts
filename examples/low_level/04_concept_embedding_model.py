"""
Concept Embedding Model
=======================

A Concept Embedding Model (CEM) represents each concept with an embedding
instead of a single number. Each concept has a positive and a negative
embedding, which are mixed according to the predicted concept probability, and
the task is predicted from the mixed embeddings. The concepts stay
interpretable and intervenable, while the embeddings carry the information a
scalar bottleneck would lose.

On the XOR toy dataset, which the linear task head of
``02_concept_bottleneck_model.py`` cannot represent, the CEM solves the task.

Reference: Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the
Accuracy-Explainability Trade-Off", NeurIPS 2022.
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import LinearEmbeddingToConcept, MixConceptEmbeddingToConcept

seed_everything(42)

# %%
# Data
# ----
dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
x = dataset.input_data
c = dataset.concepts[["C1", "C2"]]
y = dataset.concepts[["xor"]]
x_train, x_test = x[:800], x[800:]
c_train, c_test = c[:800], c[800:]
y_train, y_test = y[:800], y[800:]


# %%
# Model
# -----
# ``MixConceptEmbeddingToConcept`` splits each concept embedding into its
# positive and negative halves, mixes them with the concept probability, and
# maps the mixed embeddings to the task.
class ConceptEmbeddingModel(torch.nn.Module):
    def __init__(self, emb_size=8):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
        # one embedding per concept: (batch, 2) -> (batch, 2 concepts, emb_size)
        self.embedding_encoder = torch.nn.Sequential(
            torch.nn.Linear(16, 2 * emb_size),
            torch.nn.Unflatten(1, (2, emb_size)),
        )
        # each concept is scored from its embedding: (batch, 2, emb_size) -> (batch, 2)
        self.concept_encoder = torch.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=emb_size, out_concepts=1),
            torch.nn.Flatten(),
        )
        self.task_predictor = MixConceptEmbeddingToConcept(
            in_concepts=c.annotations,
            in_embeddings=emb_size,
            out_concepts=y.annotations,
        )

    def forward(self, x):
        embeddings = self.embedding_encoder(self.encoder(x))
        c_logits = self.concept_encoder(embeddings)
        y_logits = self.task_predictor(
            concepts=torch.sigmoid(c_logits),
            embeddings=embeddings,
        )
        return c_logits, y_logits


model = ConceptEmbeddingModel()

# %%
# Training
# --------
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
with torch.no_grad():
    c_logits, y_logits = model(x_test)
print(f"test concept accuracy: {((c_logits > 0).float() == c_test).float().mean():.2f}")
print(f"test task accuracy:    {((y_logits > 0).float() == y_test).float().mean():.2f}")
