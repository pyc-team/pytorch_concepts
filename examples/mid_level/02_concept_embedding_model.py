"""
Concept Embedding Model as a Probabilistic Model
================================================

A Concept Embedding Model (CEM) gives every concept a positive and a negative
embedding, mixes them by the concept probability, and predicts the task from
the mixed embeddings. As a probabilistic model, the
task CPD has two kinds of parents, the concepts and their embeddings::

    input -> latent -> embeddings -> concepts -> xor
                           |                      ^
                           +----------------------+

On the XOR toy dataset, where the CBM of ``01_concept_bottleneck_model.py``
fails, the CEM solves the task.

Reference: Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the
Accuracy-Explainability Trade-Off", NeurIPS 2022.
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.distributions import Delta
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    BayesianNetwork,
    DeterministicInference,
    LearnablePrior,
    LinearEmbeddingToConcept,
    MixConceptEmbeddingToConcept,
    ParametricCPD,
    Sequential,
)

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
# ``embeddings`` holds one embedding per concept, shape (2, 8). PyC's
# ``Sequential`` passes the CPD's named inputs (``embeddings=...``) to its
# first layer.
input_var = EmbeddingVariable("input", distribution=Delta, size=2)
latent = EmbeddingVariable("latent", distribution=Delta, size=16)
embeddings = EmbeddingVariable("embeddings", distribution=Delta, shape=(2, 8))
concepts = ConceptVariable("concepts", members=["C1", "C2"], distribution=Bernoulli)
xor = ConceptVariable("xor", distribution=Bernoulli)

encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
embedder = torch.nn.Sequential(
    torch.nn.Linear(16, 2 * 8),
    torch.nn.Unflatten(-1, (2, 8)),
)
# each concept is scored from its own embedding: (2, 8) -> (2,)
concept_encoder = Sequential(
    LinearEmbeddingToConcept(in_embeddings=8, out_concepts=1),
    torch.nn.Flatten(start_dim=-2),
)
task_predictor = MixConceptEmbeddingToConcept(
    in_concepts=c.annotations,
    in_embeddings=8,
    out_concepts=1,
)

model = BayesianNetwork(
    variables=[input_var, latent, embeddings, concepts, xor],
    factors=[
        ParametricCPD(input_var, parents=[], parametrization=LearnablePrior(2)),
        ParametricCPD(latent, parents=[input_var], parametrization=encoder),
        ParametricCPD(embeddings, parents=[latent], parametrization=embedder),
        ParametricCPD(
            concepts,
            parents=[embeddings],
            parametrization={"logits": concept_encoder},
        ),
        ParametricCPD(
            xor,
            parents=[concepts, embeddings],
            parametrization={"logits": task_predictor},
        ),
    ],
)
engine = DeterministicInference(model, p_int=0.5)
test_engine = DeterministicInference(model)

# %%
# Training
# --------
# With ``p_int=0.5``, each predicted concept is replaced by its target half of
# the time (random interventions, as in the CEM paper). The targets come with
# the query, as ``{name: target}``: a list of names carries none.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(500):
    optimizer.zero_grad()
    out = engine.query({"concepts": c_train, "xor": None}, evidence={"input": x_train})
    concept_loss = loss_fn(out.logits["concepts"], c_train)
    task_loss = loss_fn(out.logits["xor"], y_train)
    loss = concept_loss + 0.5 * task_loss
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
with torch.no_grad():
    out = test_engine.query(["concepts", "xor"], evidence={"input": x_test})
concept_accuracy = ((out.logits["concepts"] > 0).float() == c_test).float().mean()
task_accuracy = ((out.logits["xor"] > 0).float() == y_test).float().mean()
print(f"test concept accuracy: {concept_accuracy:.2f}")
print(f"test task accuracy:    {task_accuracy:.2f}")
