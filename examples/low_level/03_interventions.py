"""
Interventions
=============

The concepts of a concept-based model can be edited at test time, and the
task prediction follows. An expert who knows the true value of a concept can
correct the model's mistake (a *ground-truth intervention*), and anyone can ask
what the model would predict if a concept took a given value (a
*do-intervention*).

The ``intervention`` context manager overrides a layer's outputs for the
duration of a ``with`` block:

- a *strategy* gives the new values (``GroundTruthIntervention``,
  ``DoIntervention``, ...);
- a *policy* ranks the outputs, so that only the first ``quantile`` of them is
  replaced (``UniformPolicy``: no preference; ``UncertaintyInterventionPolicy``:
  least confident first).

The model is the CBM of ``02_concept_bottleneck_model.py`` with a non-linear
task head (``MLPConceptToConcept``), so that it can solve XOR.
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    DoIntervention,
    GroundTruthIntervention,
    LinearEmbeddingToConcept,
    MLPConceptToConcept,
    UncertaintyInterventionPolicy,
    UniformPolicy,
    intervention,
)

seed_everything(42)

# %%
# Data and model
# --------------
# The model sees a noisy reading of the input, so it gets some concepts wrong
# near the decision boundaries: mistakes that an expert can correct.
dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
x = dataset.input_data + 0.1 * torch.randn(1000, 2)
c = dataset.concepts[["C1", "C2"]]
y = dataset.concepts[["xor"]]
x_train, x_test = x[:800], x[800:]
c_train, c_test = c[:800], c[800:]
y_train, y_test = y[:800], y[800:]


class ConceptBottleneckModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
        self.concept_encoder = LinearEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=c.annotations,
        )
        self.task_predictor = MLPConceptToConcept(
            in_concepts=c.annotations,
            out_concepts=y.annotations,
            hidden_size=16,
        )

    def forward(self, x):
        c_logits = self.concept_encoder(embeddings=self.encoder(x))
        y_logits = self.task_predictor(concepts=torch.sigmoid(c_logits))
        return c_logits, y_logits


model = ConceptBottleneckModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(500):
    optimizer.zero_grad()
    c_logits, y_logits = model(x_train)
    loss = loss_fn(c_logits, c_train) + 0.5 * loss_fn(y_logits, y_train)
    loss.backward()
    optimizer.step()
model.eval()


def accuracy(logits, labels):
    return ((logits > 0).float() == labels).float().mean().item()


def report(title, c_logits, y_logits):
    concepts, task = accuracy(c_logits, c_test), accuracy(y_logits, y_test)
    print(f"{title:<23} concept accuracy {concepts:.3f}, task accuracy {task:.3f}")


with torch.no_grad():
    c_logits, y_logits = model(x_test)
report("no intervention:", c_logits, y_logits)

# %%
# Ground-truth interventions
# --------------------------
# The concept encoder outputs logits, so the true concepts are given as logits
# too: ``torch.logit`` maps the 0/1 labels to large negative/positive values.
c_true = torch.logit(c_test, eps=1e-6)

with torch.no_grad(), intervention(
    model.concept_encoder,
    GroundTruthIntervention(c_true),
    UniformPolicy(),
    ["C1", "C2"],
):
    c_logits, y_logits = model(x_test)
report("correct both concepts:", c_logits, y_logits)

# An expert's time is limited: correct only the least confident of the two
# concepts of each sample (``quantile=0.5``).
with torch.no_grad(), intervention(
    model.concept_encoder,
    GroundTruthIntervention(c_true),
    UncertaintyInterventionPolicy(),
    ["C1", "C2"],
    quantile=0.5,
):
    c_logits, y_logits = model(x_test)
report("correct the least sure:", c_logits, y_logits)

# %%
# Do-interventions
# ----------------
# What would the model predict if ``C1`` were true? Forcing its logit to a
# large positive value flips the XOR prediction where ``C1`` was predicted
# false, and leaves it alone where ``C1`` was already true.
with torch.no_grad():
    c_logits, y_logits = model(x_test)
    with intervention(
        model.concept_encoder,
        DoIntervention(torch.logit(torch.Tensor([1]), eps=1e-6)),
        UniformPolicy(),
        ["C1"],
    ):
        _, y_logits_do = model(x_test)
flipped = ((y_logits > 0) != (y_logits_do > 0)).squeeze(-1).float()
c1_true = c_logits[:, 0] > 0
print("do(C1 = true) flips the task prediction of")
print(f"  {flipped[~c1_true].mean():.0%} of the samples where C1 was false")
print(f"  {flipped[c1_true].mean():.0%} of the samples where C1 was true")
