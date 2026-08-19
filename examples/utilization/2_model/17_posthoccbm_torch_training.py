"""
Example: A Post-hoc Concept Bottleneck Model (PCBM) built on top of a
         pretrained black-box model with Manual PyTorch Training.

This example demonstrates how to turn an existing black-box model into a
concept bottleneck model, without ever retraining it, following the post-hoc
pipeline of Yuksekgonul et al. (ICLR 2023) with a manual PyTorch training loop.
"""

import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import CAVEmbeddingToConcept, MLP, PostHocCBM
from torchmetrics.classification import BinaryAccuracy


# Standard deviation of the noise added to the dataset's input embeddings.
INPUT_NOISE = 1.0

# The concept bank of a post-hoc CBM is rarely complete, and that is the whole
# motivation for the hybrid PCBM-h. We emulate it by dropping 'either' and its
# own parents 'lung' and 'tub'.
CONCEPT_BANK = ["asia", "smoke", "bronc", "xray"]


def add_noise(x, seed):
    """
    A noisy view of the dataset's input embeddings (see ``INPUT_NOISE``).
    """
    generator = torch.Generator().manual_seed(seed)
    return x + INPUT_NOISE * torch.randn(x.shape, generator=generator)


def train_loop(parameters, forward_fn, target, desc, n_epochs=500, lr=0.01):
    """
    A minimal manual training loop applying a BCE loss to the logits returned
    by ``forward_fn``, plus whatever extra loss term it returns alongside them.
    """
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(range(n_epochs), desc=desc, unit="epoch")
    for _ in progress_bar:
        optimizer.zero_grad()
        logits, extra_loss = forward_fn()
        loss = loss_fn(logits, target) + extra_loss
        loss.backward()
        optimizer.step()

        # Show the live loss on the progress bar instead of printing it.
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")


@torch.no_grad()
def intervention_curve(
    model,
    x,
    c,
    y,
    concept_names,
    task_names,
    low,
    high,
    n_orders=5,
):
    """
    Task accuracy as a function of how many concept scores are clamped.

    For each budget ``k`` we clamp a *random* subset of ``k`` concepts and
    average over ``n_orders`` draws — the standard random intervention policy.
    A PCBM concept is a signed *margin* rather than a probability, so an
    intervention has to name a value on that scale: ``high[i]`` stands for
    "concept present" and ``low[i]`` for "concept absent" (see Step 7).
    """
    accuracy_fn = BinaryAccuracy()
    model.eval()
    curve = []
    for budget in range(len(concept_names) + 1):
        accuracies = []
        for order in range(n_orders):
            rng = np.random.default_rng(order)
            chosen = rng.permutation(len(concept_names))[:budget]
            evidence = {'input': x}
            for i in chosen:
                evidence[concept_names[i]] = torch.where(
                    c.tensor[:, i:i + 1] > 0.5,
                    high[i],
                    low[i],
                )
            out = model(query=task_names, evidence=evidence)
            accuracies.append(accuracy_fn(
                out.params[task_names[0]]['logits'],
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
    n_features = dataset.n_features[-1]

    task_names = ["dysp"]
    concept_names = CONCEPT_BANK
    annotations = dataset.annotations.subset(concept_names + task_names)
    print(
        f"Concept bank (incomplete, as 'tub', 'lung' and 'either' are "
        f"missing): {concept_names}"
    )

    x_train = add_noise(dataset.input_data, seed=0)
    c_train = dataset.concepts[concept_names]
    y_train = dataset.concepts[task_names]

    # Split into train and test sets
    train_size = int(0.8 * len(x_train))
    x_train, x_test = x_train[:train_size], x_train[train_size:]
    c_train, c_test = c_train[:train_size], c_train[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 2: Pretrain a black-box model (no concepts involved)")
    print("=" * 60)

    latent_size = 128
    trunk = MLP(input_size=n_features, hidden_size=latent_size, n_layers=1)
    blackbox_head = nn.Linear(latent_size, 1)

    train_loop(
        parameters=(
            list(trunk.parameters()) + list(blackbox_head.parameters())
        ),
        forward_fn=lambda: (blackbox_head(trunk(x_train)), 0.0),
        target=y_train.float(),
        desc="Pretraining black box",
    )

    with torch.no_grad():
        blackbox_acc = BinaryAccuracy()(
            blackbox_head(trunk(x_test)), y_test.int()
        ).item()
    print(f"Black-box task accuracy: {blackbox_acc:.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 3: Fit CAVs post-hoc with logistic-regression probes")
    print("=" * 60)

    # Fitting the bank is what the CAV layer's ``fit`` already does: one
    # logistic-regression probe per concept on the frozen embeddings, stored as
    # unit-norm CAVs with a rescaled intercept. It hands back each probe's
    # training accuracy, the paper's check that the concept is linearly
    # readable at this layer.
    cav_bank = CAVEmbeddingToConcept(
        in_embeddings=latent_size,
        out_concepts=len(concept_names),
        C=0.1,
    )
    with torch.no_grad():
        probe_accuracies = cav_bank.fit(trunk(x_train), c_train.tensor)

    for name, accuracy in zip(concept_names, probe_accuracies):
        print(f"\tFitted CAV for {name!r} (train probe acc: {accuracy:.4f})")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Build the PostHocCBM and train its interpretable head")
    print("=" * 60)

    pcbm = PostHocCBM(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        concept_vectors=cav_bank.cavs,
        concept_intercepts=cav_bank.bias,
        residual=True,
        backbone=trunk,
        latent_size=latent_size,
    )
    print(f"PostHocCBM created successfully! ({type(pcbm).__name__})")

    pcbm.set_residual_use(False)
    pcbm.train()

    def pcbm_forward():
        out = pcbm(query=task_names, input=x_train)
        # Only the task loss plus the elastic net, as the bank is already fixed
        return out.params[task_names[0]]['logits'], pcbm.elastic_net()

    train_loop(
        parameters=[p for p in pcbm.parameters() if p.requires_grad],
        forward_fn=pcbm_forward,
        target=y_train.float(),
        desc="Training PCBM head",
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 5: Fit the PCBM-h residual sequentially")
    print("=" * 60)

    pcbm.freeze_non_residual_components()
    pcbm.set_residual_use(True)
    pcbm.train()

    train_loop(
        parameters=[p for p in pcbm.parameters() if p.requires_grad],
        forward_fn=lambda: (
            pcbm(
                query=task_names,
                input=x_train
            ).params[task_names[0]]['logits'],
            0.0,
        ),
        target=y_train.float(),
        desc="Training PCBM-h residual",
    )

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 6: Evaluation")
    print("=" * 60)

    pcbm.eval()
    with torch.no_grad():
        # A concept counts as present when its score is positive
        out = pcbm(query=concept_names, input=x_test)
        scores = out.value[concept_names]
        concept_acc = BinaryAccuracy()(
            (scores > 0).float(),
            c_test.int(),
        ).item()

        # And the task accuracy, interpretable-only (PCBM) vs hybrid (PCBM-h)
        pcbm.set_residual_use(False)
        pcbm_logits = pcbm(
            query=task_names,
            input=x_test
        ).params[task_names[0]]['logits']
        pcbm.set_residual_use(True)
        pcbm_h_logits = pcbm(
            query=task_names,
            input=x_test
        ).params[task_names[0]]['logits']

        pcbm_acc = BinaryAccuracy()(pcbm_logits, y_test.int()).item()
        pcbm_h_acc = BinaryAccuracy()(pcbm_h_logits, y_test.int()).item()

    print(f"Concept accuracy of the CAV scores: {concept_acc:.4f}\n")
    print(f"{'Model':<12} {'Task acc':>12}")
    print(f"{'Black box':<12} {blackbox_acc:>12.4f}")
    print(f"{'PCBM':<12} {pcbm_acc:>12.4f}")
    print(f"{'PCBM-h':<12} {pcbm_h_acc:>12.4f}")

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 7: Concept interventions")
    print("=" * 60)

    # Interventions clamp the concept-score variables through evidence. Unlike
    # a CBM's concept probabilities, a PCBM concept is a signed *margin* to the
    # CAV hyperplane, so an intervention has to name a value on that scale. We
    # use the recipe the original CBM paper uses for its logit bottleneck
    # (Koh et al., ICML 2020): represent a concept by the 95th percentile of
    # its empirical training scores when true and the 5th when false, which
    # states the concept firmly while keeping the clamped value inside the
    # range the task head was fitted on.
    with torch.no_grad():
        train_scores = pcbm(
            query=concept_names,
            input=x_train,
        ).value[concept_names]
    low, high = torch.quantile(
        train_scores.tensor,
        torch.tensor([0.05, 0.95]),
        dim=0,
    )

    # Note for self: the percentiles are unconditional, as in the paper, so a
    # heavily imbalanced concept can land both of them on the same side of the
    # hyperplane. 'asia' holds in 99% of this dataset, and sure enough even its
    # 5th percentile still reads as "present".
    print("Train-score percentiles per concept (5th / 95th):")
    for i, name in enumerate(concept_names):
        print(f"\t{name:<8} {low[i]:+.3f} / {high[i]:+.3f}")

    # The residual bypasses the bottleneck entirely, so no intervention can
    # reach it. That makes the interpretable configuration the one to intervene
    # on, and the PCBM/PCBM-h comparison below makes it concrete
    curves = {}
    for label, residual in (("PCBM", False), ("PCBM-h", True)):
        pcbm.set_residual_use(residual)
        curves[label] = intervention_curve(
            model=pcbm,
            x=x_test,
            c=c_test,
            y=y_test,
            concept_names=concept_names,
            task_names=task_names,
            low=low,
            high=high,
            n_orders=5,
        )
    pcbm.set_residual_use(True)

    header = "  ".join(f"k={k}" for k in range(len(concept_names) + 1))
    print("\nTask accuracy vs. #intervened concepts (random subsets)\n")
    print(f"{'Model':<12} {header}")
    for label, curve in curves.items():
        print(f"{label:<12} " + "  ".join(f"{a:.3f}" for a in curve))

    for label, curve in curves.items():
        print(
            f"\n{label}: {curve[0]:.4f} -> {curve[-1]:.4f} "
            f"(gain {curve[-1] - curve[0]:+.4f})"
        )


if __name__ == "__main__":
    main()
