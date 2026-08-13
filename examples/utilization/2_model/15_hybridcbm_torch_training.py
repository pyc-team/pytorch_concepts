"""
Example: A Hybrid Concept Bottleneck Model (HybridCBM) benchmarked against a
         standard CBM model with Manual PyTorch Training.

This example demonstrates how to initialize and train a HybridCBM
using a manual PyTorch training loop (without Lightning). We do this in a
complete and incomplete toy dataset setting, comparing the HybridCBM against a
standard CBM model. There we show how to use the HybridCBM to handle incomplete
concept annotations, while the standard CBM fails to learn effectively due to
missing concept labels.

We then look at the *other* side of that trade-off, which is the reason
Concept Embedding Models were proposed in the first place: the unsupervised
dimensions are a side channel around the bottleneck, so a Hybrid CBM responds
much more weakly to concept interventions than a plain CBM.

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
from torch_concepts.nn import \
    ConceptBottleneckModel, DeterministicInference, MLP
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy

from tqdm import tqdm

from torch_concepts.nn.modules.high.models.hybrid_cbm import \
    HybridConceptBottleneckModel


# The fraction of the concepts to keep when training incomplete models.
FRAC_SELECTED_COCNEPTS = 0.5


# Standard deviation of the noise added to the dataset's input embeddings.
# `BnLearnDataset` builds its inputs with an autoencoder over the sampled
# concept values, so a noiseless input determines every concept almost exactly
# (~100% concept accuracy) and leaves an expert nothing to correct. Degrading
# the labels will help us make concept interventions a bit more meaningful in
# the examples shown below.
INPUT_NOISE = 0.75

# Rate at which training replaces a predicted concept by its ground truth
# ("RandInt"): the task head then also learns from hard concept values, which
# is what keeps a model responsive to interventions at test time.
P_INT = 0.0


def add_noise(x, seed):
    """
    A noisy view of the dataset's input embeddings (see ``INPUT_NOISE``).
    """
    generator = torch.Generator().manual_seed(seed)
    return x + INPUT_NOISE * torch.randn(x.shape, generator=generator)


def ground_truth_for(model, ground_truth, annotations):
    """
    Re-lay the dataset ground truth in ``model``'s own annotation order.

    ``fully_observed_query`` indexes the ground truth by the model's own
    annotation. Two models here need a re-layout: the incomplete models keep
    only a subset of the labels, and a HybridCBM *extends* its annotation with
    a dummy label per unsupervised dimension. The dummy columns are never read
    (the unsupervised dimensions are embedding variables, so the query skips
    them), but the tensor still has to be that wide.
    """
    labels = list(model.concept_annotations.labels)
    known = set(annotations.labels)
    out = torch.zeros(ground_truth.shape[0], len(labels))
    for i, name in enumerate(labels):
        if name in known:
            out[:, i] = ground_truth[:, annotations.get_index(name)]
    return out


@torch.no_grad()
def intervention_curve(model, x, c, y, concept_names, task_names, n_orders=5):
    """
    Task accuracy as a function of how many concepts are intervened on.

    For each budget ``k`` we clamp a *random* subset of ``k`` concepts to their
    ground truth and average over ``n_orders`` draws — the standard random
    intervention policy. Intervened concepts are passed as ``evidence``, which
    is how PyC expresses an intervention: the variable is clamped and every
    downstream CPD consumes the clamped value.
    """
    accuracy_fn = BinaryAccuracy()
    model.eval()
    curve = []
    for budget in range(len(concept_names) + 1):
        accuracies = []
        for order in range(n_orders):
            rng = np.random.default_rng(order)
            chosen = rng.permutation(concept_names)[:budget]
            evidence = {'input': x}
            for name in chosen:
                # The engine expects raw tensors as evidence.
                evidence[name] = c[name].tensor.float()
            out = model(query=task_names, evidence=evidence)
            accuracies.append(accuracy_fn(
                out.params[task_names[0]]['logits'], y.int(),
            ).item())
        curve.append(float(np.mean(accuracies)))
    return curve


def main():

    ############################################################################
    ## Setup
    ############################################################################

    seed_everything(42)
    print(
        f"We will use {FRAC_SELECTED_COCNEPTS * 100:.1f}% of the "
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

    x_train = add_noise(complete_dataset.input_data, seed=0)
    c_train = complete_dataset.concepts[concept_names]
    y_train = complete_dataset.concepts[task_names]
    # Full concept ground truth in annotation-label order, used for training
    # with random interventions.
    gt_train = complete_dataset.concepts[list(annotations.labels)].tensor

    # Split into train and test sets
    train_size = int(0.8 * len(x_train))
    x_train, x_test = x_train[:train_size], x_train[train_size:]
    c_train, c_test = c_train[:train_size], c_train[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]
    gt_train = gt_train[:train_size]


    ############################################################################

    print("=" * 60)
    print("Step 2: Select subsample for incomplete concept annotations")
    print("=" * 60)

    # Randomly select 80% of the concepts to keep
    selected_concepts_idxs = np.random.choice(
        np.arange(len(concept_names)),
        size=int(np.ceil(FRAC_SELECTED_COCNEPTS * len(concept_names))),
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
    # are handled internally). Every model here trains with random
    # interventions at rate ``P_INT`` (if given, but by default we set this to
    # 0, so there is no RandInt training).
    train_kwargs = dict(
        train_inference=DeterministicInference,
        train_inference_kwargs={'p_int': P_INT},
    )
    complete_cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
        **train_kwargs,
    )

    print(f"Complete CBM created successfully!")
    print(f"Complete CBM type: {type(complete_cbm).__name__}")
    print(f"Complete CBM's Encoder output features: {complete_cbm.latent_size}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Initialize Incomplete CBM")
    print("=" * 60)

    # Initialize the incomplete CBM (defaults for distributions and activations
    # are handled internally). Crucially, its bottleneck must ONLY a subset of
    # the ground-truth concept set
    incomplete_annotations = annotations.subset(selected_concepts + task_names)
    incomplete_cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=incomplete_annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
        **train_kwargs,
    )

    print(f"Incomplete CBM created successfully!")
    print(f"Incomplete CBM type: {type(incomplete_cbm).__name__}")
    print(
        f"Incomplete CBM's Encoder output "
        f"features: {incomplete_cbm.latent_size}"
    )


    ############################################################################

    print("\n" + "=" * 60)
    print("Step 5: Initialize HybridCBM with incomplete concept annotations")
    print("=" * 60)

    hybrid_cbm = HybridConceptBottleneckModel(
        input_size=n_features,
        annotations=incomplete_annotations,
        task_names=task_names,
        additional_dims=(len(concept_names) - len(selected_concepts)),
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
        **train_kwargs,
    )

    print(f"Hybrid CBM created successfully!")
    print(f"Hybrid CBM type: {type(hybrid_cbm).__name__}")
    print(f"Hybrid CBM's Encoder output features: {hybrid_cbm.latent_size}")

    complete_hybrid_cbm = HybridConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        additional_dims=8,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,
        **train_kwargs,
    )


    ############################################################################


    print("\n" + "=" * 60)
    print("Step 6: Training loop with torch loss")
    print("=" * 60)

    n_epochs = 500
    loss_fn = nn.BCEWithLogitsLoss()

    models = [complete_cbm, incomplete_cbm, hybrid_cbm, complete_hybrid_cbm]
    names = [
        "Complete CBM",
        "Incomplete CBM",
        "Hybrid CBM",
        "Complete Hybrid CBM",
    ]
    all_concepts = np.arange(len(concept_names))
    concept_idxs = [
        all_concepts,
        selected_concepts_idxs,
        selected_concepts_idxs,
        all_concepts,
    ]

    for model, name, selected_idxs in zip(models, names, concept_idxs):
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.train()
        progress_bar = tqdm(
            range(n_epochs),
            desc=f"Training {name}",
            unit="epoch",
        )
        model_concepts = [concept_names[i] for i in selected_idxs]
        supervised = model_concepts + task_names
        # Concatenate concepts and tasks as target
        target = c_train[model_concepts].union_with(y_train).float()
        # Passing the ground truth as the query (instead of a bare list of
        # names) is what enables the engine's ``p_int`` random interventions.
        # The params in the output are still the model's own predictions, so
        # the supervised loss below is unaffected.
        query = model.fully_observed_query(
            ground_truth_for(model, gt_train, annotations)
        )
        for _ in progress_bar:
            opt.zero_grad()

            # Forward pass - query all variables (concepts + tasks)
            out = model(query=query, input=x_train)

            # Compute loss on all outputs
            loss = loss_fn(out.logits[supervised], target)

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

    for model, name, selected_idxs in zip(models, names, concept_idxs):
        print(f"\nEvaluating {name}...")
        model.eval()
        model_concepts = [concept_names[i] for i in selected_idxs]
        query = model_concepts + task_names
        with torch.no_grad():
            out = model(query=query, input=x_test)

            # Compute accuracy using BinaryAccuracy
            concept_acc = concept_acc_fn(
                out.logits[model_concepts],
                c_test[model_concepts].int(),
            ).item()
            task_acc = task_acc_fn(out.logits[task_names], y_test.int()).item()

            print(f"\tConcept accuracy: {concept_acc:.4f}")
            print(f"\tTask accuracy: {task_acc:.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 8: Concept interventions — the cost of the extra capacity")
    print("=" * 60)

    curves = {
        name: intervention_curve(
            model=model,
            x=x_test,
            c=c_test,
            y=y_test,
            concept_names=concept_names,
            task_names=task_names,
            n_orders=5,
        )
        for model, name in (
            (complete_cbm, "Complete CBM"),
            (complete_hybrid_cbm, "Complete Hybrid CBM"),
        )
    }

    header = "  ".join(f"k={k}" for k in range(len(concept_names) + 1))
    print(f"\nTask accuracy vs. #intervened concepts (random subsets)\n")
    print(f"{'Model':<22} {header}")
    for name, curve in curves.items():
        print(f"{name:<22} " + "  ".join(f"{a:.3f}" for a in curve))

    for name, curve in curves.items():
        print(
            f"\n{name}: {curve[0]:.4f} -> {curve[-1]:.4f} "
            f"(gain {curve[-1] - curve[0]:+.4f})"
        )

if __name__ == "__main__":
    main()
