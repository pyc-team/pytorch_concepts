"""
Example: A Post-hoc Concept Bottleneck Model (PCBM) built on top of a
         pretrained black-box model with Manual PyTorch Training.

This example demonstrates the full post-hoc pipeline of Yuksekgonul et al.
("Post-hoc Concept Bottleneck Models", ICLR 2023,
https://arxiv.org/abs/2205.15480):

1. pretrain a black-box model end-to-end on the task (no concepts involved);
2. freeze its trunk and fit one concept-activation vector (CAV) per concept
   *post-hoc*, with logistic-regression probes on the trunk's embeddings;
3. build a PostHocCBM from the frozen trunk and the fitted CAVs, and train
   only its sparse (elastic-net regularised) interpretable head;
4. fit the hybrid PCBM-h residual sequentially: freeze everything but the
   residual head and let it recover the black box's accuracy;
5. compare the black box, the interpretable PCBM, and the hybrid PCBM-h, and
   intervene on the concept scores.

The model uses:
- a PostHocCBM (with ``residual=True`` for the PCBM-h variant)
- lightning=False (default) for pure PyTorch module behavior
- Manual optimizer and loss function setup
- Annotations for concept metadata
"""

import numpy as np
import torch
from torch import nn

from sklearn.linear_model import LogisticRegression

from torch_concepts import seed_everything
from torch_concepts.nn import MLP, PostHocCBM
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy

from tqdm import tqdm


def train_loop(parameters, forward_fn, target, desc, n_epochs=500, lr=0.01):
    """Minimal manual training loop: BCE on the logits of ``forward_fn``."""
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(range(n_epochs), desc=desc, unit="epoch")
    for _ in progress_bar:
        optimizer.zero_grad()
        logits, extra_loss = forward_fn()
        loss = loss_fn(logits, target) + extra_loss
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")


def main():

    seed_everything(42)

    ############################################################################

    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy dataset")
    print("=" * 60)

    dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    n_features = dataset.n_features[-1]

    task_names = ["dysp"]
    # The concept bank of a post-hoc CBM is rarely complete — that is the
    # motivation for the hybrid PCBM-h. We emulate this with a very
    # incomplete bank containing only distant ancestors of the task
    # ('smoke' influences 'dysp' only through the hidden 'bronc'/'lung'):
    # the interpretable bottleneck then loses information that the residual
    # has to recover.
    concept_names = ["asia", "smoke"]
    annotations = dataset.annotations.subset(concept_names + task_names)
    print(f"Concept bank (incomplete): {concept_names}")

    x_train = dataset.input_data
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

    # Any pretrained model works here; we train a small MLP trunk + linear
    # head end-to-end on the task only.
    latent_size = 128
    trunk = MLP(input_size=n_features, hidden_size=latent_size, n_layers=1)
    blackbox_head = nn.Linear(latent_size, 1)

    train_loop(
        parameters=list(trunk.parameters()) + list(blackbox_head.parameters()),
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

    # One logistic-regression probe per concept, fitted on the *frozen*
    # trunk's embeddings — the (coefficients, intercept) of each probe are the
    # concept's activation vector, as in the original PCBM pipeline.
    with torch.no_grad():
        embeddings = trunk(x_train).numpy()

    concept_vectors, concept_intercepts = [], []
    for i, name in enumerate(concept_names):
        probe = LogisticRegression(max_iter=1000, C=0.1)
        probe.fit(embeddings, c_train.tensor[:, i].numpy())
        concept_vectors.append(probe.coef_[0])
        concept_intercepts.append(probe.intercept_[0])
        print(f"\tFitted CAV for {name!r} "
              f"(train probe acc: {probe.score(embeddings, c_train.tensor[:, i].numpy()):.4f})")
    concept_vectors = torch.tensor(np.stack(concept_vectors)).float()
    concept_intercepts = torch.tensor(np.stack(concept_intercepts)).float()

    ############################################################################

    print("\n" + "=" * 60)
    print("Step 4: Build the PostHocCBM and train its interpretable head")
    print("=" * 60)

    # The trunk enters as the (frozen) backbone and the fitted CAVs as the
    # (frozen) concept bank; ``residual=True`` prepares the PCBM-h variant,
    # whose residual we keep disabled while fitting the interpretable head.
    pcbm = PostHocCBM(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        concept_vectors=concept_vectors,
        concept_intercepts=concept_intercepts,
        residual=True,
        backbone=trunk,
        latent_size=latent_size,
    )
    print(f"PostHocCBM created successfully! ({type(pcbm).__name__})")

    pcbm.set_residual_use(False)
    pcbm.train()

    def pcbm_forward():
        out = pcbm(query=task_names, input=x_train)
        # Only the task loss + the elastic-net sparsity regulariser: the
        # concept bank is fixed, so no concept supervision is needed.
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

    # The paper's recipe: freeze the backbone, the CAVs and the interpretable
    # head, re-enable the residual, and let it recover whatever accuracy the
    # concept bottleneck lost.
    pcbm.freeze_non_residual_components()
    pcbm.set_residual_use(True)
    pcbm.train()

    train_loop(
        parameters=[p for p in pcbm.parameters() if p.requires_grad],
        forward_fn=lambda: (
            pcbm(query=task_names, input=x_train)
            .params[task_names[0]]['logits'],
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
        # Concept accuracy of the post-hoc bottleneck: a concept is predicted
        # present when its score (signed distance to the CAV hyperplane) > 0.
        out = pcbm(query=concept_names, input=x_test)
        scores = torch.cat(
            [out.params[name]['value'] for name in concept_names],
            dim=1,
        )
        concept_acc = BinaryAccuracy()(
            (scores > 0).float(), c_test.int()
        ).item()

        # Task accuracy: interpretable-only (PCBM) vs hybrid (PCBM-h).
        pcbm.set_residual_use(False)
        pcbm_logits = pcbm(query=task_names, input=x_test) \
            .params[task_names[0]]['logits']
        pcbm.set_residual_use(True)
        pcbm_h_logits = pcbm(query=task_names, input=x_test) \
            .params[task_names[0]]['logits']

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

    # Interventions clamp the concept-score variables through evidence; here
    # we use +/-1 scores from the ground truth (the reference implementation's
    # active/inactive intervention values). The residual pathway bypasses the
    # bottleneck, so we intervene on the interpretable configuration.
    evidence = {'input': x_test}
    for i, name in enumerate(concept_names):
        evidence[name] = (2.0 * c_test.tensor[:, i:i + 1] - 1.0).float()

    pcbm.set_residual_use(False)
    with torch.no_grad():
        out = pcbm(query=task_names, evidence=evidence)
        task_acc_int = BinaryAccuracy()(
            out.params[task_names[0]]['logits'], y_test.int()
        ).item()
    pcbm.set_residual_use(True)

    print(f"PCBM task accuracy without interventions: {pcbm_acc:.4f}")
    print(f"PCBM task accuracy with GT interventions: {task_acc_int:.4f}")


if __name__ == "__main__":
    main()
