"""
Concept-based Memory Reasoner
=============================

A Concept-based Memory Reasoner (CMR) predicts the task with a logic rule
picked from a learned memory. The memory decodes, for every rule, the role each
concept plays in it: positive literal, negative literal, or irrelevant. A
selector weighs the rules per sample, and the task probability is the selected
rules evaluated on the concept probabilities.

``RuleMemory`` holds the rules and ``RuleConceptEmbeddingToConcept`` evaluates
them. The same layer gives CMR's two task paths: at ``rec_weight=0`` a rule is
scored on the task alone, above ``0`` also on how well it reconstructs the
concepts. The task loss is label-switched: negative labels are scored on the
first path, positive labels on the second.

Reference: Debot et al., "Interpretable Concept-Based Memory Reasoning",
NeurIPS 2024.
"""

# %%
import torch

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    LinearEmbeddingToConcept,
    RuleConceptEmbeddingToConcept,
    RuleMemory,
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
# The rule layers read the selector weights and the rule roles packed into one
# embedding: ``n_rules`` weights plus ``n_rules * n_concepts * 3`` roles per task.
class ConceptMemoryReasoner(torch.nn.Module):
    def __init__(self, n_rules=10, memory_size=100, rec_weight=0.1):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.LeakyReLU())
        self.concept_encoder = LinearEmbeddingToConcept(
            in_embeddings=10,
            out_concepts=c.annotations,
        )
        # rule selector logits: (batch, 1 task, n_rules)
        self.selector = torch.nn.Sequential(
            MLP(input_size=10, hidden_size=10, output_size=n_rules),
            torch.nn.Unflatten(-1, (1, n_rules)),
        )
        # role probabilities: (1 task, n_rules, 2 concepts, 3 roles)
        self.memory = RuleMemory(
            n_tasks=1,
            n_rules=n_rules,
            n_concepts=2,
            latent_size=memory_size,
        )
        rule_size = n_rules * (1 + 2 * 3)
        self.task_predictor = RuleConceptEmbeddingToConcept(
            out_concepts=y.annotations,
            in_concepts=c.annotations,
            in_embeddings=rule_size,
            n_rules=n_rules,
            rec_weight=0.0,
        )
        self.rec_predictor = RuleConceptEmbeddingToConcept(
            out_concepts=y.annotations,
            in_concepts=c.annotations,
            in_embeddings=rule_size,
            n_rules=n_rules,
            rec_weight=rec_weight,
        )

    def forward(self, x):
        h = self.encoder(x)
        c_logits = self.concept_encoder(embeddings=h)
        c_probs = torch.sigmoid(c_logits)
        selector = torch.softmax(self.selector(h), dim=-1)
        roles = self.memory()
        roles = roles.expand(len(x), *roles.shape)
        rules = torch.cat([selector.flatten(1), roles.flatten(1)], dim=-1)
        # both rule layers return probabilities, not logits
        y_probs = self.task_predictor(concepts=c_probs, embeddings=rules)
        y_probs_rec = self.rec_predictor(concepts=c_probs, embeddings=rules)
        return c_logits, y_probs, y_probs_rec


model = ConceptMemoryReasoner()

# %%
# Training
# --------
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
concept_loss_fn = torch.nn.BCEWithLogitsLoss()
task_loss_fn = torch.nn.BCELoss(reduction="none")
for epoch in range(500):
    optimizer.zero_grad()
    c_logits, y_probs, y_probs_rec = model(x_train)
    task_loss = torch.where(
        y_train == 1,
        task_loss_fn(y_probs_rec, y_train),
        task_loss_fn(y_probs, y_train),
    )
    loss = concept_loss_fn(c_logits, c_train) + task_loss.mean()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f}")

# %%
# Evaluation
# ----------
model.eval()
with torch.no_grad():
    c_logits, y_probs, _ = model(x_test)
print(f"test concept accuracy: {((c_logits > 0).float() == c_test).float().mean():.2f}")
print(f"test task accuracy:    {((y_probs > 0.5).float() == y_test).float().mean():.2f}")
