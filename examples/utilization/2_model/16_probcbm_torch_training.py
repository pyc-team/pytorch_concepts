"""
Example: A Probabilistic Concept Bottleneck Model (ProbCBM) benchmarked
         against a standard CBM model with Manual PyTorch Training.

This example demonstrates how to initialize and train a ProbCBM (Kim et al.,
ICML 2023) using a manual PyTorch training loop (without Lightning).

IMPORTANT NOTE: for simplicity, this example trains the ProbCBM *jointly*, in
a single pass over one combined loss. The paper does not: it fits the concept
predictor first and the class predictor second, holds the backbone frozen for a
warm-up, uses two learning rates with a cosine schedule, and replaces predicted
concept embeddings by their ground-truth anchors (``p_replace``) while the
class predictor trains. None of that is done here. To reproduce the paper's
pipeline, build the model with ``lightning=True``, which carries the whole
recipe, and hand it to a Trainer as shown in example 16.5.

The script show how to use:
- a ProbCBM and a ConceptBottleneckModel (both with pure PyTorch behavior,
  i.e. ``lightning=False``)
- Manual optimizer and loss function setup for ProbCBM
- per-concept uncertainty from the embedding variances
- Monte-Carlo class uncertainty by swapping in ``AncestralSamplingInference``
- concept interventions, which in a ProbCBM replace the predicted embeddings
  with the learned concept anchors (``anchor_embeddings``)
"""

import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import (
    AncestralSamplingInference, ConceptBottleneckModel, DeterministicInference
)
from torch_concepts.nn import MLP, ProbCBM
from torchmetrics.classification import BinaryAccuracy


# Standard deviation of the noise added to the dataset's input embeddings.
INPUT_NOISE = 1.0

# Rate at which the CBM replaces a predicted concept by its ground truth.
P_INT = 0.5

# Weight of ProbCBM's variational information bottleneck regulariser.
VIB_BETA = 0.00005


def add_noise(x, seed):
    """
    A noisy view of the dataset's input embeddings (see ``INPUT_NOISE``).
    """
    generator = torch.Generator().manual_seed(seed)
    return x + INPUT_NOISE * torch.randn(x.shape, generator=generator)


def train_model(
    model,
    query,
    supervised,
    x_train,
    target,
    n_epochs=500,
    lr=0.01,
):
    """
    The manual PyTorch training loop shared by both models: one joint loss over
    the ``supervised`` concepts and tasks, plus ProbCBM's VIB regulariser.

    ``query`` is what the model is asked for and ``supervised`` what the loss
    scores, which are not the same thing: the CBM's query carries ground truth
    so the engine can teacher-force, and the ProbCBM's also carries the
    embedding variables so the VIB term can read their moments.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    progress_bar = tqdm(
        range(n_epochs),
        desc=f"Training {type(model).__name__}",
        unit="epoch",
    )
    for _ in progress_bar:
        optimizer.zero_grad()
        out = model(query=query, input=x_train)

        loss = loss_fn(out.logits[supervised], target[supervised])
        if isinstance(model, ProbCBM):
            loss = loss + VIB_BETA * model.vib_kl(out)

        loss.backward()
        optimizer.step()

        # Show the live loss on the progress bar instead of printing it.
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return model


def evaluate_model(model, concept_names, task_names, x, c, y):
    """
    The concept and task accuracy of a trained model.
    """
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        out = model(query=(concept_names + task_names), input=x)
        concept_acc = concept_acc_fn(out.logits[concept_names], c.int()).item()
        task_acc = task_acc_fn(out.logits[task_names], y.int()).item()
    return concept_acc, task_acc


def concept_evidence(model, x, c, chosen):
    """
    Evidence clamping each of the ``chosen`` concept variables to its ground
    truth, which is how one intervenes on a standard CBM.
    """
    # The engine expects raw tensors as evidence.
    return {name: c[name].tensor.float() for name in chosen}


def anchor_evidence(model, x, c, chosen):
    """
    Evidence clamping the ``chosen`` concepts' *embeddings* to their
    ground-truth anchors, which is how one intervenes on ProbCBMs.
    Concepts left out keep the embedding the model predicted for them, which is
    why we need a forward pass first.
    """
    out = model(query=model.embedding_query_names, input=x)
    return model.anchor_embeddings(
        {name: c[name].tensor.float() for name in chosen},
        out=out,
    )


@torch.no_grad()
def intervention_curve(model, x, c, y, concept_names, task_names, evidence_fn):
    """
    Task accuracy as a function of how many concepts are intervened on.

    For each budget ``k`` we clamp a *random* subset of ``k`` concepts to their
    ground truth and average over a few draws — the standard random
    intervention policy. ``evidence_fn`` builds the actual evidence, since a
    CBM and a ProbCBM are intervened on different variables.
    """
    accuracy_fn = BinaryAccuracy()
    model.eval()
    curve = []
    for budget in range(len(concept_names) + 1):
        accuracies = []
        for order in range(5):
            rng = np.random.default_rng(order)
            chosen = rng.permutation(concept_names)[:budget]
            evidence = {'input': x, **evidence_fn(model, x, c, chosen)}
            out = model(query=task_names, evidence=evidence)
            accuracies.append(accuracy_fn(
                out.logits[task_names],
                y.int(),
            ).item())
        curve.append(float(np.mean(accuracies)))
    return curve


def main():

    ############################################################################
    ## Setup
    ############################################################################

    seed_everything(42)

    ############################################################################

    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy dataset")
    print("=" * 60)

    dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    annotations = dataset.annotations
    n_features = dataset.n_features[-1]

    task_names = ["dysp"]
    concept_names = [
        n for n in dataset.concept_names if n not in task_names
    ]
    supervised = concept_names + task_names

    x_train = add_noise(dataset.input_data, seed=0)
    c_train = dataset.concepts[concept_names]
    y_train = dataset.concepts[task_names]
    # Full concept ground truth, used by the CBM's random interventions
    gt_train = dataset.concepts[list(annotations.labels)].tensor

    # Split into train and test sets
    train_size = int(0.8 * len(x_train))
    x_train, x_test = x_train[:train_size], x_train[train_size:]
    c_train, c_test = c_train[:train_size], c_train[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]
    gt_train = gt_train[:train_size]

    target = c_train.union_with(y_train).float()

    ############################################################################

    # Init the different models
    print("\n" + "=" * 60)
    print("Step 2: Initialize standard CBM")
    print("=" * 60)

    # Standard CBM baseline
    cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
        train_inference=DeterministicInference,
        train_inference_kwargs={'p_int': P_INT},
    )

    print(f"CBM created successfully!")
    print(f"CBM type: {type(cbm).__name__}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 3: Initialize ProbCBM")
    print("=" * 60)

    # Note that training uses ``AncestralSamplingInference`` so the concept
    # embeddings are *sampled* (reparameterised) during training, as in the
    # paper. Evaluation instead keeps the default ``DeterministicInference``,
    # which is the paper's sampling-free evaluation.
    #
    # We set ``p_int=0`` because this example trains jointly: forcing an
    # embedding would also decide the concept decoded from it, so a joint
    # concept loss would be handed its own answer. The paper avoids that by
    # forcing only while the class predictor trains alone (see example 16.5).
    prob_cbm = ProbCBM(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        embedding_size=16,        # Size of each probabilistic concept embedding
        class_embedding_size=32,  # Size of the class-embedding space
        train_inference=AncestralSamplingInference,
        train_inference_kwargs={'p_int': 0.0},
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"ProbCBM created successfully!")
    print(f"ProbCBM type: {type(prob_cbm).__name__}")
    print(
        f"Probabilistic embedding variables: "
        f"{prob_cbm.embedding_query_names}"
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Training loop with torch loss")
    print("=" * 60)

    # The CBM trains with random interventions on its concepts, so its task
    # head learns to read hard concept values too.
    train_model(
        model=cbm,
        query=cbm.fully_observed_query(gt_train),
        supervised=supervised,
        x_train=x_train,
        target=target,
    )

    # The ProbCBM query also includes the embedding variables so that the VIB
    # regulariser can be computed from their (loc, scale) parameters.
    train_model(
        model=prob_cbm,
        query=supervised + prob_cbm.embedding_query_names,
        supervised=supervised,
        x_train=x_train,
        target=target,
    )

    ############################################################################

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    cbm_accs = evaluate_model(
        model=cbm,
        concept_names=concept_names,
        task_names=task_names,
        x=x_test,
        c=c_test,
        y=y_test,
    )
    prob_accs = evaluate_model(
        model=prob_cbm,
        concept_names=concept_names,
        task_names=task_names,
        x=x_test,
        c=c_test,
        y=y_test,
    )

    print(f"{'Model':<12} {'Concept acc':>12} {'Task acc':>12}")
    print(f"{'CBM':<12} {cbm_accs[0]:>12.4f} {cbm_accs[1]:>12.4f}")
    print(f"{'ProbCBM':<12} {prob_accs[0]:>12.4f} {prob_accs[1]:>12.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 6: Concept uncertainty from the embedding variances")
    print("=" * 60)

    prob_cbm.eval()
    with torch.no_grad():
        out = prob_cbm(query=prob_cbm.embedding_query_names, input=x_test)
        uncertainty = prob_cbm.concept_uncertainty(out).mean(dim=0)

    print("Average per-concept uncertainty on the test set:")
    for name, unc in zip(concept_names, uncertainty):
        print(f"\t{name:<8} {unc.item():.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 7: Monte-Carlo class uncertainty via ancestral sampling")
    print("=" * 60)

    # Swapping the evaluation inference engine means the embeddings are now
    # sampled from their Normal distributions (and the concepts from relaxed
    # Bernoullis), so repeated forward passes yield Monte-Carlo estimates whose
    # spread is the class uncertainty *derived from* the concept uncertainty.
    prob_cbm.setup_inference(inference=AncestralSamplingInference)
    prob_cbm.eval()

    n_mc_samples = 20
    with torch.no_grad():
        mc_probs = torch.stack([
            torch.sigmoid(
                prob_cbm(query=task_names, input=x_test).logits[task_names]
            )
            for _ in range(n_mc_samples)
        ])
    print(f"MC task probability (n={n_mc_samples} samples):")
    print(
        f"\tmean of per-sample stds: "
        f"{mc_probs.std(dim=0).mean().item():.4f}"
    )

    # And restore the deterministic engine for the intervention demo below.
    prob_cbm.setup_inference(inference=DeterministicInference)

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 8: Concept interventions")
    print("=" * 60)

    # A CBM is intervened on its concept variables, while a ProbCBM has its
    # concept embeddings clamped to the ground-truth anchors
    curves = {
        name: intervention_curve(
            model=model,
            x=x_test,
            c=c_test,
            y=y_test,
            concept_names=concept_names,
            task_names=task_names,
            evidence_fn=evidence_fn,
        )
        for model, name, evidence_fn in (
            (cbm, 'CBM', concept_evidence),
            (prob_cbm, 'ProbCBM', anchor_evidence),
        )
    }

    header = "  ".join(f"k={k}" for k in range(len(concept_names) + 1))
    print("\nTask accuracy vs. #intervened concepts (random subsets)\n")
    print(f"{'Model':<12} {header}")
    for name, curve in curves.items():
        print(f"{name:<12} " + "  ".join(f"{a:.3f}" for a in curve))

    for name, curve in curves.items():
        print(
            f"\n{name}: {curve[0]:.4f} -> {curve[-1]:.4f} "
            f"(gain {curve[-1] - curve[0]:+.4f})"
        )


if __name__ == "__main__":
    main()
