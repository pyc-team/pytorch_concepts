"""
Example: A Post-hoc Concept Bottleneck Model (PCBM / PCBM-h) trained with
         PyTorch Lightning.

This example demonstrates how to train a PCBM (Yuksekgonul et al., ICLR 2023)
with Lightning rather than by hand. Passing ``lightning=True`` turns the model
into a ``LightningModule`` that already carries the paper's training recipe, so
a plain ``Trainer.fit`` runs both of its stages. Example 17 is the same
pipeline driven by a manual loop.
"""

import torch

from pytorch_lightning import Trainer
from torch import nn

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.nn import CAVEmbeddingToConcept, MLP, PostHocCBM
from torchmetrics.classification import BinaryAccuracy


# The concept bank of a post-hoc CBM is rarely complete, and that is the whole
# motivation for the hybrid PCBM-h. We emulate it by dropping 'either' and its
# own parents 'lung' and 'tub'.
CONCEPT_BANK = ["asia", "smoke", "bronc", "xray"]


def pretrain_blackbox(datamodule, n_features, latent_size, n_epochs=200):
    """
    Train a small black box end-to-end on the task alone, with no concepts,
    and hand back its (about to be frozen) trunk.
    """
    trunk = MLP(input_size=n_features, hidden_size=latent_size, n_layers=1)
    head = nn.Linear(latent_size, 1)
    optimizer = torch.optim.AdamW(
        list(trunk.parameters()) + list(head.parameters()),
        lr=0.01,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(n_epochs):
        for batch in datamodule.train_dataloader():
            optimizer.zero_grad()
            logits = head(trunk(batch['inputs']['x']))
            loss_fn(logits, batch['concepts']['c'][:, -1:].float()).backward()
            optimizer.step()
    return trunk


def fit_cavs(trunk, datamodule, concept_index):
    """
    One logistic-regression probe per concept on the *frozen* trunk's
    embeddings, which is that concept's CAV. The probes are fitted by the CAV
    layer itself, which leaves them unit-norm with a rescaled intercept.
    """
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            embeddings.append(trunk(batch['inputs']['x']))
            labels.append(batch['concepts']['c'])
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)[:, concept_index]

    # ``fit`` is the CAV layer's own post-hoc probe fitting, so the bank it
    # leaves behind is exactly what the PCBM wants handed to it.
    cav_bank = CAVEmbeddingToConcept(
        in_embeddings=embeddings.shape[-1],
        out_concepts=len(concept_index),
        C=0.1,
    )
    cav_bank.fit(embeddings, labels)
    return cav_bank.cavs, cav_bank.bias


@torch.no_grad()
def task_accuracy(model, datamodule, task_names, use_residual):
    """
    Task accuracy over the test set, with the residual on or off.
    """
    model.eval()
    model.set_residual_use(use_residual)
    accuracy_fn = BinaryAccuracy()
    for batch in datamodule.test_dataloader():
        out = model(query=task_names, input=batch['inputs']['x'])
        target = model.prepare_target(batch['concepts']['c'])
        accuracy_fn.update(out.logits[task_names], target[task_names].int())
    return accuracy_fn.compute().item()


def main():

    ############################################################################
    ## Setup
    ############################################################################

    seed_everything(42)
    latent_size = 128

    ############################################################################

    print("=" * 60)
    print("Step 1: Generate toy dataset")
    print("=" * 60)

    dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    datamodule = ConceptDataModule(
        dataset=dataset,
        batch_size=256,
        val_size=0.1,
        test_size=0.2,
        seed=42,
    )
    datamodule.setup()

    annotations = dataset.annotations
    n_features = dataset.input_data.shape[1]
    task_names = ["dysp"]
    concept_names = CONCEPT_BANK
    bank_annotations = annotations.subset(concept_names + task_names)
    concept_index = [annotations.get_index(n) for n in concept_names]
    print(f"Concept bank (incomplete): {concept_names} | task: {task_names}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 2: Pretrain a black box, then fit CAVs on its frozen trunk")
    print("=" * 60)

    trunk = pretrain_blackbox(datamodule, n_features, latent_size)
    concept_vectors, concept_intercepts = fit_cavs(
        trunk,
        datamodule,
        concept_index,
    )
    print(
        f"Fitted {len(concept_names)} CAVs of size "
        f"{concept_vectors.shape[1]}"
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 3: Initialize a PostHocCBM that trains itself")
    print("=" * 60)

    model = PostHocCBM(
        input_size=n_features,
        annotations=bank_annotations,
        task_names=task_names,
        concept_vectors=concept_vectors,
        concept_intercepts=concept_intercepts,
        residual=True,
        backbone=trunk,
        latent_size=latent_size,
        # --- the paper's training recipe ---
        reg_strength=1e-5,
        l1_ratio=0.99,
        interpretable_epochs=30,
        residual_epochs=20,
        residual_l2_penalty=0.001,
        lr=0.01,
        lightning=True,
    )

    print(f"PostHocCBM created successfully! ({type(model).__name__})")
    print(
        f"Stages: interpretable x{model.interpretable_epochs}, "
        f"residual x{model.residual_epochs} ({model.total_epochs} epochs)"
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Fit with a plain Lightning Trainer")
    print("=" * 60)

    trainer = Trainer(
        max_epochs=model.total_epochs,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    pcbm_acc = task_accuracy(
        model,
        datamodule,
        task_names,
        use_residual=False,
    )
    pcbm_h_acc = task_accuracy(
        model,
        datamodule,
        task_names,
        use_residual=True,
    )

    print(f"{'Model':<12} {'Task acc':>12}")
    print(f"{'PCBM':<12} {pcbm_acc:>12.4f}")
    print(f"{'PCBM-h':<12} {pcbm_h_acc:>12.4f}")
    print(
        "\nNote: the recovery is complete here because the dataset's inputs "
        "are built from\nthe concept values, the task included, so the latent "
        "the residual reads encodes\n'dysp' almost directly."
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 6: The interpretable head one can actually read")
    print("=" * 60)

    weights = model._interpretable_heads[0].weight.detach()[0]
    for name, weight in sorted(
        zip(concept_names, weights.tolist()),
        key=lambda p: -abs(p[1]),
    ):
        print(f"\t{name:<8} {weight:+.4f}")


if __name__ == "__main__":
    main()
