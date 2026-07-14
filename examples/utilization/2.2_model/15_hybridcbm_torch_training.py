"""
Example: A Hybrid Concept Bottleneck Model (HybridCBM) benchmarked against a
         standard CBM model with Manual PyTorch Training.

This example demonstrates how to initialize and train a HybridCBM
using a manual PyTorch training loop (without Lightning). We do this in a
complete and incomplete toy dataset setting, comparing the HybridCBM against a
standard CBM model. There we show how to use the HybridCBM to handle incomplete
concept annotations, while the standard CBM fails to learn effectively due to
missing concept labels.

The model uses:
- a HybridConceptBottleneckModel and a ConceptBottleneckModel
- lightning=False (default) for pure PyTorch module behavior
- Manual optimizer and loss function setup
- Annotations for concept metadata
- Compares both the HybridCBM and standard CBM models.
"""

import numpy as np
import torch
from torch import nn

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, MLP
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy

from tqdm import tqdm

from torch_concepts.nn.modules.high.models.hybrid_cbm import HybridConceptBottleneckModel



def main():

    ############################################################################
    ## Setup
    ############################################################################

    fraction_incomplete_concepts = 0.25
    seed_everything(42)
    print(
        f"We will use {fraction_incomplete_concepts * 100:.1f}% of the "
        f"concepts when training incomplete models."
    )

    ############################################################################

    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate complete toy dataset")
    print("=" * 60)

    complete_dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    annotations = complete_dataset.annotations
    n_features = complete_dataset.n_features[-1]

    task_names = ["dysp"]
    concept_names = [
        n for n in complete_dataset.concept_names if n not in task_names
    ]

    x_train = complete_dataset.input_data
    c_train = complete_dataset.concepts[concept_names]
    y_train = complete_dataset.concepts[task_names]

    # Split into train and test sets
    train_size = int(0.8 * len(x_train))
    x_train, x_test = x_train[:train_size], x_train[train_size:]
    c_train, c_test = c_train[:train_size], c_train[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    ############################################################################

    print("=" * 60)
    print("Step 2: Select subsample for incomplete concept annotations")
    print("=" * 60)

    # Randomly select 80% of the concepts to keep
    selected_concepts_idxs = np.random.choice(
        np.arange(len(concept_names)),
        size=int(np.ceil(fraction_incomplete_concepts * len(concept_names))),
        replace=False,
    )
    selected_concepts = [concept_names[i] for i in selected_concepts_idxs]
    print(f"Selected concepts for incomplete annotations: {selected_concepts}")
    print(
        f"\tThis means we selected {len(selected_concepts)} out "
        f"of {len(concept_names)} concepts."
    )

    ############################################################################

    # Init the different models
    print("\n" + "=" * 60)
    print("Step 3: Initialize Complete CBM")
    print("=" * 60)

    # Initialize the complete CBM (defaults for distributions and activations
    # are handled internally)
    complete_cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"Complete CBM created successfully!")
    print(f"Complete CBM type: {type(complete_cbm).__name__}")
    print(f"Complete CBM's Encoder output features: {complete_cbm.latent_size}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Initialize Incomplete CBM")
    print("=" * 60)

    # Initialize the incomplete CBM (defaults for distributions and activations
    # are handled internally). Crucially, its bottleneck must ONLY contain the
    # selected concepts: we subset the annotations so the task predictor has no
    # access to the dropped concepts (directly or as free latent neurons).
    incomplete_annotations = annotations.subset(selected_concepts + task_names)
    incomplete_cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=incomplete_annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"Incomplete CBM created successfully!")
    print(f"Incomplete CBM type: {type(incomplete_cbm).__name__}")
    print(f"Incomplete CBM's Encoder output features: {incomplete_cbm.latent_size}")


    ############################################################################

    print("\n" + "=" * 60)
    print("Step 5: Initialize HybridCBM with incomplete concept annotations")
    print("=" * 60)

    # The HybridCBM sees the SAME incomplete concept set as the incomplete CBM,
    # but compensates with unsupervised bottleneck dimensions that can recover
    # whatever the dropped concepts would have carried.
    hybrid_cbm = HybridConceptBottleneckModel(
        input_size=n_features,
        annotations=incomplete_annotations,
        task_names=task_names,
        additional_dims=(len(concept_names) - len(selected_concepts)),
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"Hybrid CBM created successfully!")
    print(f"Hybrid CBM type: {type(hybrid_cbm).__name__}")
    print(f"Hybrid CBM's Encoder output features: {hybrid_cbm.latent_size}")


    ############################################################################


    print("\n" + "=" * 60)
    print("Step 6: Training loop with torch loss")
    print("=" * 60)

    n_epochs = 500
    hybrid_optimizer = torch.optim.AdamW(
        hybrid_cbm.parameters(),
        lr=0.01,
    )
    comp_cbm_optimizer = torch.optim.AdamW(
        complete_cbm.parameters(),
        lr=0.01,
    )
    incomp_cbm_optimizer = torch.optim.AdamW(
        incomplete_cbm.parameters(),
        lr=0.01,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for model, opt, name, selected_idxs in zip(
        [complete_cbm, incomplete_cbm, hybrid_cbm],
        [comp_cbm_optimizer, incomp_cbm_optimizer, hybrid_optimizer],
        ["Complete CBM", "Incomplete CBM", "Hybrid CBM"],
        [
            np.arange(len(concept_names)),
            selected_concepts_idxs,
            selected_concepts_idxs,
        ]
    ):
        model.train()
        progress_bar = tqdm(
            range(n_epochs),
            desc=f"Training {name}",
            unit="epoch",
        )
        selected_concepts = [concept_names[i] for i in selected_idxs]
        query = selected_concepts + task_names
        for _ in progress_bar:
            opt.zero_grad()

            # Concatenate concepts and tasks as target
            target = c_train[selected_concepts].union_with(y_train).float()

            # Forward pass - query all variables (concepts + tasks)
            out = model(query=query, input=x_train)

            # Compute loss on all outputs
            logits = torch.cat(
                [out.params[name]['logits'] for name in query],
                dim=1,
            )
            loss = loss_fn(logits, target)

            loss.backward()
            opt.step()

            # Show the live loss on the progress bar instead of printing it.
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    ############################################################################

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 7: Evaluation")
    print("=" * 60)

    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    for model, name, selected_idxs in zip(
        [complete_cbm, incomplete_cbm, hybrid_cbm],
        ["Complete CBM", "Incomplete CBM", "Hybrid CBM"],
        [
            np.arange(len(concept_names)),
            selected_concepts_idxs,
            selected_concepts_idxs,
        ]
    ):
        print(f"\nEvaluating {name}...")
        model.eval()
        selected_concepts = [concept_names[i] for i in selected_idxs]
        query = selected_concepts + task_names
        with torch.no_grad():
            out = model(query=query, input=x_test)
            c_pred = torch.cat(
                [out.params[name]['logits'] for name in selected_concepts],
                dim=1,
            )
            y_pred = torch.cat(
                [out.params[name]['logits'] for name in task_names],
                dim=1,
            )

            # Compute accuracy using BinaryAccuracy
            concept_acc = concept_acc_fn(
                c_pred,
                c_test[selected_concepts].int(),
            ).item()
            task_acc = task_acc_fn(y_pred, y_test.int()).item()

            print(f"\tConcept accuracy: {concept_acc:.4f}")
            print(f"\tTask accuracy: {task_acc:.4f}")

if __name__ == "__main__":
    main()
