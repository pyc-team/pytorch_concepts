"""
Example: Concept Memory Reasoner with the High-Level API

The Concept Memory Reasoner (CMR) predicts each task with a *learned rulebook*
instead of a black-box head.  A learnable memory decodes, for every task and
every rule, the role each concept plays in that rule (positive literal,
negative literal, or irrelevant).  A per-sample selector then picks which rule
to apply, so the prediction is a logic rule the model can show you.

Two things set CMR apart from the other models in this folder:

- It reports **probabilities**, not logits: its rule layers compute a
  probability by construction (``param_for_discrete_var = "probs"``), so the
  concept loss must be built on ``torch.nn.BCELoss`` with
  ``binary_param="probs"`` rather than ``BCEWithLogitsLoss``.
- Its task loss is **label-switched**: negative labels are scored with the
  ordinary rule prediction, positive labels with the reconstruction-aware one.
  ``CMRTaskLoss`` implements this, and ``CompositeLoss`` combines it with the
  concept term.

The last step reads the learned rules back out of the model.
"""

import torch
from torch.distributions import Bernoulli
from torchmetrics.classification import BinaryAccuracy
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.nn import (
    CMRTaskLoss,
    CompositeLoss,
    ConceptLoss,
    ConceptMemoryReasoner,
    ConceptSubset,
    DeterministicInference,
    MLP,
)
from torch_concepts.data import BnLearnDataModule


def evaluate(model, datamodule, concept_names, task_names):
    """Evaluate on the test set and return concept/task accuracy."""
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            # CMR reports probabilities: no sigmoid needed.
            out = model(input=batch['inputs']['x'],
                        query=concept_names + task_names)
            target = batch['concepts']['c']
            n_concepts = len(concept_names)

            concept_acc_fn(out.probs[concept_names], target[:, :n_concepts].int())
            task_acc_fn(out.probs[task_names], target[:, n_concepts:].int())

    concept_acc = concept_acc_fn.compute().item()
    task_acc = task_acc_fn.compute().item()
    print(f"Concept accuracy: {concept_acc:.4f}")
    print(f"Task accuracy: {task_acc:.4f}")
    return concept_acc, task_acc


def print_rulebook(model, datamodule, concept_names, task_names, n_rules):
    """Decode the learned memory into human-readable logic rules.

    ``rule_roles`` is a categorical over three roles per concept: index 0 means
    the concept appears as a positive literal, 1 as a negated literal, and 2
    that the rule ignores it.  With ``hard_roles_at_eval=True`` the eval-mode
    roles are one-hot, so the argmax is the rule itself.
    """
    role_symbols = ['{}', 'not {}', None]

    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    with torch.no_grad():
        out = model(input=batch['inputs']['x'],
                    query=['rule_selector', 'rule_roles'])

    n_tasks = len(task_names)
    n_concepts = len(concept_names)
    # Both come out flattened over the batch; the memory is sample-independent.
    roles = out.probs['rule_roles'].unflatten(
        -1, (n_tasks, n_rules, n_concepts, 3)
    )[0]
    selector = out.logits['rule_selector'].unflatten(
        -1, (n_tasks, n_rules)
    ).softmax(dim=-1)
    usage = selector.mean(dim=0)

    for t, task in enumerate(task_names):
        print(f"\nRules for '{task}' (mean selector weight in brackets):")
        for r in range(n_rules):
            literals = [
                role_symbols[role].format(concept)
                for concept, role in zip(concept_names,
                                         roles[t, r].argmax(dim=-1).tolist())
                if role_symbols[role] is not None
            ]
            body = ' and '.join(literals) if literals else 'True'
            print(f"  [{usage[t, r]:.2f}] {task} <- {body}")


def main():
    seed = 42
    seed_everything(seed)

    # =========================================================================
    # STEP 1: DATA
    # =========================================================================
    print("=" * 60)
    print("Step 1: Generate bnlearn dataset")
    print("=" * 60)

    n_samples = 10000
    batch_size = 2048
    datamodule = BnLearnDataModule(
        seed=seed,
        generation_seed=seed,
        name='asia',
        n_gen=n_samples,
        batch_size=batch_size,
        val_size=0.1,
        test_size=0.2,
    )
    datamodule.setup()
    annotations = datamodule.annotations

    n_features = datamodule.input_data.shape[1]
    task_names = ['dysp']
    concept_names = [n for n in annotations.labels if n not in task_names]
    n_rules = 10

    print(f"Input features: {n_features}")
    print(f"Concepts: {len(concept_names)} - {concept_names}")
    print(f"Tasks: {len(task_names)} - {task_names}")
    print(f"Training samples: {n_samples}")

    # =========================================================================
    # STEP 2: THE CMR LOSS
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Build the CMR loss")
    print("=" * 60)

    loss = CompositeLoss(
        terms=[
            ConceptSubset(
                ConceptLoss(
                    binary=torch.nn.BCELoss(),
                    binary_param="probs"
                ),
                names=concept_names,
            ),
            CMRTaskLoss(task_names),
        ],
        weights=[1.0, 1.0],
        names=["concepts", "tasks"],
    )

    # =========================================================================
    # STEP 3: MODEL AND TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Train the ConceptMemoryReasoner")
    print("=" * 60)

    model = ConceptMemoryReasoner(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=32, n_layers=2),
        latent_size=32,
        n_rules=n_rules,
        memory_latent_size=64,
        memory_decoder_hidden_layers=1,
        selector_hidden_layers=1,
        # Crisp one-hot roles at eval time, so the rules can be read off.
        hard_roles_at_eval=True,
        rec_weight=1.0,
        inference=DeterministicInference,
        train_inference=DeterministicInference,
        lightning=True,
        loss=loss,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.01},
    )
    print(f"Model type: {type(model).__name__}")
    print(f"Rules per task: {model.n_rules}")

    trainer = Trainer(max_epochs=200)
    trainer.fit(model, datamodule=datamodule)

    # =========================================================================
    # STEP 4: EVALUATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    evaluate(model, datamodule, concept_names, task_names)

    # =========================================================================
    # STEP 5: READ THE LEARNED RULES
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Inspect the learned rulebook")
    print("=" * 60)
    print_rulebook(model, datamodule, concept_names, task_names, n_rules)


if __name__ == "__main__":
    main()
