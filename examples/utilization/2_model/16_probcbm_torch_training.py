"""
Example: A Probabilistic Concept Bottleneck Model (ProbCBM) benchmarked
         against a standard CBM model with Manual PyTorch Training.

This example demonstrates how to initialize and train a ProbCBM
(Kim et al., "Probabilistic Concept Bottleneck Models", ICML 2023,
https://arxiv.org/abs/2306.01574) using a manual PyTorch training loop
(without Lightning), and how to use its probabilistic machinery:

- the VIB regulariser on the probabilistic concept embeddings
  (``model.vib_kl``);
- per-concept uncertainty estimates from the embedding variances
  (``model.concept_uncertainty``);
- Monte-Carlo class uncertainty by swapping the inference engine to
  ``AncestralSamplingInference``;
- concept interventions (clamping concepts to ground truth), which in a
  ProbCBM replace the predicted embeddings with the learned concept anchors.

The models use:
- a ProbCBM and a ConceptBottleneckModel
- lightning=False (default) for pure PyTorch module behavior
- Manual optimizer and loss function setup
- Annotations for concept metadata
"""

import torch
from torch import nn

from torch_concepts import seed_everything
from torch_concepts.nn import (
    AncestralSamplingInference,
    ConceptBottleneckModel,
    DeterministicInference,
    MLP,
    ProbCBM,
)
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy

from tqdm import tqdm


def train_model(model, query, x_train, target, vib_beta=0.0,
                vib_query_names=None, teacher_force_gt=None, n_epochs=500,
                lr=0.01):
    """Manual PyTorch training loop shared by both models.

    For the ProbCBM, ``vib_query_names`` contains the embedding variables
    (``model.embedding_query_names``), which are added to the query so the
    VIB KL regulariser can be computed; the supervised loss is only applied
    to the concept/task logits.

    When ``teacher_force_gt`` is given (the integer-coded concept ground
    truth, columns in ``model.concept_annotations.labels`` order), it is
    supplied with the query so the training engine can teacher-force the
    concept values at its ``p_int`` rate — the paper's ``intervention_prob``:
    the class head then also trains on ground-truth anchor embeddings, which
    makes it robust to test-time interventions.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    vib_query_names = vib_query_names or []
    if teacher_force_gt is not None:
        # ``build_query`` maps the ground truth to {variable_name: tensor},
        # which is what the engine's teacher forcing consumes (uniformly for
        # plate and individual layouts).
        forced_query = model.build_query(teacher_force_gt)
        full_query = {**forced_query, **{n: None for n in vib_query_names}}
        logit_names = list(forced_query)
    else:
        full_query = query + vib_query_names
        logit_names = query

    model.train()
    progress_bar = tqdm(
        range(n_epochs),
        desc=f"Training {type(model).__name__}",
        unit="epoch",
    )
    for _ in progress_bar:
        optimizer.zero_grad()

        # Forward pass - query all supervised variables (concepts + tasks),
        # plus the embedding variables when the VIB regulariser is used.
        out = model(query=full_query, input=x_train)

        # Compute the supervised loss on all concept/task outputs
        logits = torch.cat(
            [out.params[name]['logits'] for name in logit_names],
            dim=1,
        )
        loss = loss_fn(logits, target)

        # ProbCBM's variational information bottleneck regulariser
        if vib_beta > 0:
            loss = loss + vib_beta * model.vib_kl(out)

        loss.backward()
        optimizer.step()

        # Show the live loss on the progress bar instead of printing it.
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return model


def evaluate_model(model, concept_names, task_names, x, c, y):
    """Concept and task accuracy of a trained model."""
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        out = model(query=(concept_names + task_names), input=x)
        c_pred = torch.cat(
            [out.params[name]['logits'] for name in concept_names],
            dim=1,
        )
        y_pred = torch.cat(
            [out.params[name]['logits'] for name in task_names],
            dim=1,
        )
        concept_acc = concept_acc_fn(c_pred, c.int()).item()
        task_acc = task_acc_fn(y_pred, y.int()).item()
    return concept_acc, task_acc


def main():

    ############################################################################
    ## Setup
    ############################################################################

    vib_beta = 0.00005  # weight of ProbCBM's VIB regulariser
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
    query = concept_names + task_names

    x_train = dataset.input_data
    c_train = dataset.concepts[concept_names]
    y_train = dataset.concepts[task_names]
    # Full concept ground truth in annotation-label order, used for
    # teacher forcing (the paper's ``intervention_prob``).
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

    # Standard CBM baseline (defaults for distributions and activations are
    # handled internally)
    cbm = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"CBM created successfully!")
    print(f"CBM type: {type(cbm).__name__}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 3: Initialize ProbCBM")
    print("=" * 60)

    # ProbCBM: each concept is represented by a Gaussian embedding whose mean
    # and standard deviation are predicted from the input; concepts are
    # decoded from distances to learnable positive/negative anchors, and the
    # task from distances to learnable class anchors.
    #
    # Training uses ``AncestralSamplingInference`` so the concept embeddings
    # are *sampled* (reparameterised) during training, as in the paper: this
    # is what gives the embedding variances a learning signal beyond the VIB
    # regulariser. ``p_int=0.5`` is the paper's ``intervention_prob``: when
    # the training query carries the ground truth, predicted concept values
    # are replaced by it half of the time (teacher forcing), so the task head
    # also learns from ground-truth anchor embeddings. Evaluation keeps the
    # default ``DeterministicInference`` (the paper's sampling-free
    # evaluation, which propagates the means).
    prob_cbm = ProbCBM(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        embedding_size=16,        # size of each probabilistic concept embedding
        class_embedding_size=32,  # size of the class-embedding space
        train_inference=AncestralSamplingInference,
        train_inference_kwargs={'p_int': 0.5},
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
    )

    print(f"ProbCBM created successfully!")
    print(f"ProbCBM type: {type(prob_cbm).__name__}")
    print(f"Probabilistic embedding variables: {prob_cbm.embedding_query_names}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Training loop with torch loss")
    print("=" * 60)

    train_model(cbm, query, x_train, target)
    # The ProbCBM query also includes the embedding variables so the VIB KL
    # regulariser can be computed from their (loc, scale) parameters.
    train_model(
        prob_cbm,
        query,
        x_train,
        target,
        vib_beta=vib_beta,
        vib_query_names=prob_cbm.embedding_query_names,
        teacher_force_gt=gt_train,
    )

    ############################################################################

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    cbm_accs = evaluate_model(
        cbm, concept_names, task_names, x_test, c_test, y_test,
    )
    prob_accs = evaluate_model(
        prob_cbm, concept_names, task_names, x_test, c_test, y_test,
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
        out = prob_cbm(
            query=prob_cbm.embedding_query_names,
            input=x_test,
        )
        uncertainty = prob_cbm.concept_uncertainty(out).mean(dim=0)

    print("Average per-concept uncertainty on the test set:")
    for name, unc in zip(concept_names, uncertainty):
        print(f"\t{name:<8} {unc.item():.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 7: Monte-Carlo class uncertainty via ancestral sampling")
    print("=" * 60)

    # Swap the evaluation inference engine: embeddings are now sampled from
    # their Normal distributions (and concepts from relaxed Bernoullis), so
    # repeated forward passes yield Monte-Carlo estimates whose spread is the
    # class uncertainty *derived from the concept uncertainty*.
    prob_cbm.setup_inference(inference=AncestralSamplingInference)
    prob_cbm.eval()

    n_mc_samples = 20
    with torch.no_grad():
        mc_probs = torch.stack([
            torch.sigmoid(
                prob_cbm(query=task_names, input=x_test)
                .params[task_names[0]]['logits']
            )
            for _ in range(n_mc_samples)
        ])
    print(f"MC task probability (n={n_mc_samples} samples):")
    print(f"\tmean of per-sample stds: {mc_probs.std(dim=0).mean().item():.4f}")

    # Restore the deterministic engine for the intervention demo below.
    prob_cbm.setup_inference(inference=DeterministicInference)

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 8: Concept interventions")
    print("=" * 60)

    # Clamping concepts to their ground truth replaces the predicted concept
    # embeddings with the learned ground-truth anchors (the ProbCBM
    # intervention semantics). Intervened concepts are passed as *evidence*,
    # so only the task is queried.
    evidence = {'input': x_test}
    for name in concept_names:
        # The engine expects raw tensors as evidence.
        evidence[name] = c_test[name].tensor.float()

    prob_cbm.eval()
    with torch.no_grad():
        out = prob_cbm(query=task_names, evidence=evidence)
        y_pred = torch.cat(
            [out.params[name]['logits'] for name in task_names],
            dim=1,
        )
        task_acc_int = BinaryAccuracy()(y_pred, y_test.int()).item()

    print(f"ProbCBM task accuracy without interventions: {prob_accs[1]:.4f}")
    print(f"ProbCBM task accuracy with GT interventions: {task_acc_int:.4f}")
    print(
        "\nNote: on this toy dataset the intervened accuracy is the ceiling "
        "achievable\nfrom the true concepts ('dysp' is inherently noisy given "
        "its parents), while\nthe free-running accuracy is higher because the "
        "input embedding encodes all\nconcepts — soft concept probabilities "
        "leak task information that hard\nground-truth interventions remove."
    )


if __name__ == "__main__":
    main()
