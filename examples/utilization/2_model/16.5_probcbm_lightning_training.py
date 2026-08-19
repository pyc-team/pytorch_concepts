"""
Example: A Probabilistic Concept Bottleneck Model (ProbCBM) trained with
         PyTorch Lightning.

This example demonstrates how to train a ProbCBM (Kim et al., ICML 2023) with
Lightning rather than by hand. Passing ``lightning=True`` turns the model into
a ``LightningModule`` that already carries the paper's training recipe, so a
plain ``Trainer.fit`` reproduces it. Example 16 is the same model driven by a
manual loop.

The script show how to use:
- a ProbCBM with ``lightning=True``, which brings its own loss (Eq. 7), its
  sequential concept-then-class stages, the backbone warm-up, and ``p_replace``
- a ConceptDataModule for the data plumbing
- ``AncestralSamplingInference`` during training, so the embeddings are sampled
- per-concept uncertainty from the embedding variances
"""

import torch

from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.nn import AncestralSamplingInference, MLP, ProbCBM
from torchmetrics.classification import BinaryAccuracy


# Gradient clipping is the one part of the recipe that belongs to the Trainer
# rather than the model, since Lightning owns the optimisation loop.
CLIP_GRAD_MAX_NORM = 2.0


@torch.no_grad()
def evaluate(model, datamodule, concept_names, task_names):
    """
    The concept and task accuracy of a trained model, over the test set.
    """
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    for batch in datamodule.test_dataloader():
        out = model(
            query=concept_names + task_names,
            input=batch['inputs']['x'],
        )
        target = model.prepare_target(batch['concepts']['c'])
        concept_acc_fn.update(
            out.logits[concept_names], target[concept_names].int(),
        )
        task_acc_fn.update(out.logits[task_names], target[task_names].int())
    return concept_acc_fn.compute().item(), task_acc_fn.compute().item()


def main():

    ############################################################################
    ## Setup
    ############################################################################

    seed_everything(42)

    ############################################################################

    print("=" * 60)
    print("Step 1: Generate toy dataset")
    print("=" * 60)

    # The same Bayesian network as example 16, so the two are comparable.
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
    concept_names = [n for n in annotations.labels if n not in task_names]
    print(f"Concepts: {concept_names} | task: {task_names}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 2: Initialize a ProbCBM that trains itself")
    print("=" * 60)

    # As in example 16, training still uses ``AncestralSamplingInference`` so
    # the concept embeddings are sampled, which is what gives their variances a
    # gradient beyond the VIB term
    model = ProbCBM(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        embedding_size=16,        # Size of each probabilistic concept embedding
        class_embedding_size=32,  # Size of the class-embedding space
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,
        train_inference=AncestralSamplingInference,
        # --- From here onwards are the args for the lightning component ---
        vib_beta=0.00005,         # Eq. 7's lambda_KL
        intervention_prob=0.5,    # p_replace, applied in the class stage
        train_class_mode='sequential',
        concept_epochs=30,
        class_epochs=15,
        warm_epochs=3,
        lr=0.001,
        lr_ratio=10.0,
        lightning=True,
    )

    print(f"ProbCBM created successfully! ({type(model).__name__})")
    print(
        f"Stages: concept x{model.concept_epochs}, "
        f"class x{model.class_epochs} ({model.total_epochs} epochs)"
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 3: Fit with a plain Lightning Trainer")
    print("=" * 60)

    # ``total_epochs`` covers both stages, and the model switches between them
    # on its own at the right epoch
    trainer = Trainer(
        max_epochs=model.total_epochs,
        gradient_clip_val=CLIP_GRAD_MAX_NORM,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)

    # We expect the evaluation of this model to be better than the ProbCBM
    # trained by hand in example 16, because the Lightning model does both
    # pre-training and better management of the optimizers.
    concept_acc, task_acc = evaluate(
        model, datamodule, concept_names, task_names,
    )
    print(f"Concept accuracy: {concept_acc:.4f}")
    print(f"Task accuracy: {task_acc:.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 5: Concept uncertainty from the embedding variances")
    print("=" * 60)

    # The probabilistic machinery is untouched by how the model was trained,
    # so the uncertainty estimates read exactly as in example 16
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    with torch.no_grad():
        out = model(
            query=model.embedding_query_names,
            input=batch['inputs']['x'],
        )
        uncertainty = model.concept_uncertainty(out).mean(dim=0)

    print("Average per-concept uncertainty on a test batch:")
    for name, unc in zip(concept_names, uncertainty):
        print(f"\t{name:<8} {unc.item():.4f}")


if __name__ == "__main__":
    main()
