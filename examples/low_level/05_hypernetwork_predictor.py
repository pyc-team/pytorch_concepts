"""
Rules from a Memory
===================

A hypernetwork predictor writes the task head anew for every sample: the task
logit is a linear rule over the concept probabilities, and the rule's weights
are generated from an embedding. If that embedding is picked from a small
learned memory, the model's whole decision logic is a handful of rules that can
be read off, and every prediction uses exactly one of them.

``SelectorEmbeddingEncoder`` holds the memory and picks one slot per sample;
``HyperlinearConceptEmbeddingToConcept`` turns the picked slot into the weights
of a rule. No single linear rule over ``C1`` and ``C2`` computes XOR, but two
rules, each used on part of the input space, do.

References: Debot et al., "Interpretable Concept-Based Memory Reasoning",
NeurIPS 2024 (rules selected from a memory); De Felice et al., "Causally
Reliable Concept Bottleneck Models", NeurIPS 2025 (the hypernetwork predictor).
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    HyperlinearConceptEmbeddingToConcept,
    LinearEmbeddingToConcept,
    SelectorEmbeddingEncoder,
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
class RuleMemoryModel(torch.nn.Module):
    def __init__(self, n_rules=4, rule_size=8):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU())
        self.concept_encoder = LinearEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=c.annotations,
        )
        # memory of rule embeddings; one slot per sample: (batch, 1 task, rule_size)
        self.selector = SelectorEmbeddingEncoder(
            in_features=16,
            out_features=rule_size,
            memory_size=n_rules,
        )
        # rule embedding -> weights of a linear rule over the 2 concepts
        self.task_predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=2,
            in_embeddings=rule_size,
            hidden_size=16,
        )

    def forward(self, x, sampling=False):
        h = self.encoder(x)
        c_logits = self.concept_encoder(embeddings=h)
        rule = self.selector(h, sampling=sampling)
        y_logits = self.task_predictor(
            concepts=torch.sigmoid(c_logits),
            embeddings=rule,
        )
        return c_logits, y_logits


model = RuleMemoryModel()

# %%
# Training
# --------
# ``sampling=True`` picks one slot per sample (straight-through Gumbel-softmax),
# so that each rule has to work on its own.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(1000):
    optimizer.zero_grad()
    c_logits, y_logits = model(x_train, sampling=True)
    loss = loss_fn(c_logits, c_train) + 0.5 * loss_fn(y_logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
model.eval()
with torch.no_grad():
    c_logits, y_logits = model(x_test)
print(f"test concept accuracy: {((c_logits > 0).float() == c_test).float().mean():.2f}")
print(f"test task accuracy:    {((y_logits > 0).float() == y_test).float().mean():.2f}")

# %%
# Reading the rules
# -----------------
# Each memory slot is a rule: the hypernetwork maps it to one weight per
# concept. Which slot a sample uses is the selector's most likely choice.
with torch.no_grad():
    rule_size = model.selector.out_features
    slots = model.selector.memory.weight.view(-1, rule_size)  # (n_rules, rule_size)
    weights = model.task_predictor.hypernet(slots)  # (n_rules, 2)
    used = model.selector.selector(model.encoder(x_test)).argmax(-1).flatten()
bias = model.task_predictor.bias_mean.item()
for rule in used.unique().tolist():
    w1, w2 = weights[rule].tolist()
    print(f"rule {rule}: logit(xor) = {w1:+.1f} C1 {w2:+.1f} C2 {bias:+.1f}")
for c1, c2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    mask = (c_test[:, 0] == c1) & (c_test[:, 1] == c2)
    rules, counts = used[mask].unique(return_counts=True)
    shares = ", ".join(
        f"rule {r} {n / mask.sum():.0%}"
        for r, n in zip(rules.tolist(), counts.tolist())
    )
    print(f"C1={c1}, C2={c2} uses {shares}")
