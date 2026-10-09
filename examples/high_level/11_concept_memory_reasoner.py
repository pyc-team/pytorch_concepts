"""
Concept-based Memory Reasoner
=============================

A Concept-based Memory Reasoner (CMR) predicts each task with a rule picked
from a learned rulebook instead of a black-box head. A memory decodes, for
every task and rule, the role each concept plays in the rule (positive literal,
negative literal, or irrelevant), and a selector picks the rule per sample, so
every prediction comes with the rule that made it.

Two things set CMR apart from the other models:

- It reports ``probs``, not ``logits``: its rule layers compute a probability
  by construction, so the concept term uses ``BCELoss``.
- Its task loss is label-switched: negative labels are scored on the ordinary
  rule prediction, positive labels on the reconstruction-aware one, which the
  model reports as ``tasks_with_rec``. ``CMRTaskLoss`` implements it.

Data: the ASIA network (see ``mid_level/05_learning_a_bayesian_network.py``),
with ``dysp`` as the task and the other variables as concepts.

Reference: Debot et al., "Interpretable Concept-Based Memory Reasoning",
NeurIPS 2024.
"""

# %%
import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    CMRTaskLoss,
    CompositeLoss,
    ConceptLoss,
    ConceptMemoryReasoner,
    ConceptMetrics,
    ConceptSubset
)

seed_everything(42)

# %%
# Data
# ----
datamodule = BnLearnDataModule(
    name="asia",
    seed=42,
    batch_size=2048,
    root=str(DATA_ROOT / "asia"),
)
task_names = ["dysp"]

# %%
# Loss
# ----
loss = CompositeLoss(
    terms=[
        ConceptSubset(ConceptLoss(binary=torch.nn.BCELoss()), exclude=task_names),
        CMRTaskLoss(task_names),
    ],
    names=["concepts", "tasks"],
)

# %%
# Model and training
# ------------------
# ``hard_roles_at_eval=True`` makes the roles one-hot in evaluation, so the
# rules can be read off.
n_rules = 10
model = ConceptMemoryReasoner(
    input_size=datamodule.n_features[-1],
    annotations=datamodule.annotations,
    task_names=task_names,
    backbone=MLP(datamodule.n_features[-1], 32, n_layers=2),
    latent_size=32,
    n_rules=n_rules,
    memory_latent_size=64,
    rec_weight=1.0,
    hard_roles_at_eval=True,
    lightning=True,
    loss=loss,
    metrics=ConceptMetrics(
        annotations=datamodule.annotations,
        binary={"accuracy": BinaryAccuracy},
        per_concept=True,
    ),
    optim_class=torch.optim.AdamW,
    optim_kwargs={"lr": 0.01},
)
trainer = Trainer(max_epochs=200, logger=False, enable_checkpointing=False)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

# %%
# Reading the rules
# -----------------
# ``rule_roles`` holds, per task, rule and concept, the probabilities of the
# three roles; their argmax is the rule. ``rule_selector`` gives how often each
# rule is used on the test set.
concepts = model.intermediate_concept_names
x_test = datamodule.dataset.input_data[datamodule.testset.indices]
model.eval()
with torch.no_grad():
    out = model(query=["rule_selector", "rule_roles"], input=x_test)
roles = out.probs["rule_roles"].tensor[0].view(n_rules, len(concepts), 3)
usage = out.logits["rule_selector"].tensor.softmax(-1).mean(0).tolist()
literal = ["{}", "not {}"]
for r, rule in enumerate(roles.argmax(-1).tolist()):
    body = " and ".join(literal[k].format(n) for n, k in zip(concepts, rule) if k < 2)
    print(f"[{usage[r]:.2f}] dysp <- {body or 'True'}")
