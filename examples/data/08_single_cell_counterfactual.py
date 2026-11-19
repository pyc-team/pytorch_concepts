"""
Single-cell Counterfactual Experiment
=====================================

Investigate the following counterfactual question:
How would gene expression change in cells of type 2 if concept_0 were set to 1?

We synthetically generate two paired single-cell datasets:
- Original dataset: Gene expression profiles are generated based on the original concept values of each cell.
- Counterfactual dataset: Gene expression profiles are generated for the same cells, 
using the same underlying configuration, but with concept_0 forced to 1 for cells of type 2. 
All other cells retain their original concept values.

The goal is to train a model on the original data, intervene on its
predicted concept values (setting concept_0 to 1 for cells of type 2), and compare the resulting gene expression
predictions against the counterfactual dataset as ground truth.

For this experiment, we construct a custom train/test split to prevent data leakage and enable counterfactual evaluation. 
Specifically, all cells of type 2 with concept_0 = 1 are excluded from the dataset splits. 
Half of the type 2 cells with concept_0 = 0 are assigned to the test set, while the remaining eligible cells are used for training. 
No validation set is created.

The split is computed explicitly below and passed to the datamodule through
a FixedIndicesSplitter. The intervened dataset contains the same cells in the same order.
We use the test indices to select the paired counterfactual targets.

Optional annotation experiments are selected with ANNOTATION_MODIFICATION:
- noise: flip binary labels with probability p_noise in n_mod concepts;
- drop: remove n_mod concept columns from supervision;
- add: append n_mod independent Bernoulli concepts with probability p_active;
- duplicate: append n_replica copies of each of n_mod selected concepts.
As in scCBGM main.py, concept_0 is protected. Gene expression, the original
labels and the paired counterfactual targets stay fixed. Modified annotations
are selected with the existing set_generated_concepts(use_as_gt=True) API.

This experiment is adapted from the original implementation accompanying the paper scCBGM: Single-Cell Editing via Concept Bottlenecks (Andersson et al., ICML 2026).
"""

import numpy as np
import torch
from torch import nn

from torch_concepts import AnnotatedTensor, Annotations, seed_everything
from torch_concepts.data import SingleCellDataModule
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    CompositeLoss,
    ConceptBottleneckVAE,
    ConceptLoss,
    DeterministicInference,
    DoIntervention,
    KLDivergenceLoss,
    MSEReconstructionLoss,
    OrthogonalityLoss,
    UniformPolicy,
    intervention,
)


def counterfactual_indices(dataset, concept, cell_type, seed=42):
    """scCBGM split for setting one binary concept to 1 in the chosen cell type."""
    cell_mask = dataset.cell_types.numpy() == cell_type
    active = dataset.native_concepts[concept].tensor.numpy().ravel() == 1
    controls = np.flatnonzero(cell_mask & ~active)
    if len(controls) < 2:
        raise ValueError("At least two eligible control cells are required.")
    test_idxs = np.random.default_rng(seed).permutation(controls)[:len(controls) // 2]
    train_mask = ~(cell_mask & active)
    train_mask[test_idxs] = False
    return np.flatnonzero(train_mask).tolist(), test_idxs.tolist()


def modify_annotations(
    concepts, mode="identity", n_mod=1, *, target_concept="concept_0",
    p_noise=0.5, p_active=0.5, n_replica=1, seed=42,
):
    """scCBGM annotation ablations, keeping the intervention target unchanged.

    n_mod is a count, a fraction of the non-target concepts, or "all" for noise.
    Fractions use the original round/clamp rule. Missing annotations mean whole
    columns removed, rather than missing entries in individual cells.
    Duplicate columns have unique names, including when n_replica > 1.
    """
    if mode not in {"identity", "noise", "drop", "add", "duplicate"}:
        raise ValueError("mode must be identity, noise, drop, add or duplicate.")
    if mode == "identity":
        return concepts
    target = concepts[[target_concept]]
    labels = [name for name in concepts.annotations.labels if name != target_concept]
    observed = concepts[concepts.annotations.subset(labels)]
    n_concepts = len(labels)
    if isinstance(n_mod, str):
        if n_mod != "all" or mode != "noise":
            raise ValueError('n_mod="all" is supported only for noise.')
        n_mod = n_concepts
    elif 0 < n_mod < 1:
        n_mod = min(max(int(np.round(n_concepts * n_mod)), 1), n_concepts - 1)
    if n_mod < 0 or int(n_mod) != n_mod or (mode != "add" and n_mod > n_concepts):
        raise ValueError("Invalid number of non-target concepts to modify.")
    n_mod = int(n_mod)
    if n_mod == 0:
        return observed.union_with(target)
    rng = np.random.default_rng(seed)
    if mode != "add":
        selected = rng.choice(n_concepts, size=n_mod, replace=False).tolist()
    if mode == "drop":
        keep = [name for i, name in enumerate(labels) if i not in selected]
        observed = observed[observed.annotations.subset(keep)]
    elif mode == "noise":
        observed = observed.clone()
        values = observed.tensor[:, selected]
        mask = torch.as_tensor(
            rng.binomial(1, p_noise, size=values.shape),
            dtype=torch.bool, device=values.device,
        )
        observed.tensor[:, selected] = torch.where(mask, 1 - values, values)
    elif mode == "add":
        probabilities = np.atleast_1d(p_active)
        if len(probabilities) == 1:
            probabilities = np.repeat(probabilities, n_mod)
        elif len(probabilities) != n_mod:
            probabilities = rng.choice(probabilities, size=n_mod, replace=True)
        values = np.column_stack([
            rng.binomial(1, p, size=len(concepts)) for p in probabilities
        ])
        annotations = Annotations(
            labels=[f"added_{i}" for i in range(n_mod)],
            cardinalities=[1] * n_mod, types=["binary"] * n_mod,
        )
        added = AnnotatedTensor(
            torch.as_tensor(values, dtype=concepts.dtype, device=concepts.device),
            annotations, axis=concepts.axis,
        )
        observed = observed.union_with(added)
    elif mode == "duplicate":
        if n_replica < 1 or int(n_replica) != n_replica:
            raise ValueError("n_replica must be a positive integer.")
        for i in selected:
            original = observed[[labels[i]]]
            for replica in range(int(n_replica)):
                annotations = original.annotations.rename({labels[i]: f"duplicate_{i}_{replica}"})
                observed = observed.union_with(
                    AnnotatedTensor(original.tensor, annotations, axis=concepts.axis)
                )
    return observed.union_with(target)


seed_everything(42)
EPOCHS = 20
# Change only this dictionary to choose the annotation experiment:
ANNOTATION_MODIFICATION = {"mode": "identity"}
# {"mode": "noise", "n_mod": "all", "p_noise": 0.5}
# {"mode": "drop", "n_mod": 1}
# {"mode": "add", "n_mod": 1, "p_active": 0.5}
# {"mode": "duplicate", "n_mod": 1, "n_replica": 1}


# 1. Create the original dataset.
dm = SingleCellDataModule(
    root=DATA_ROOT / "single_cell_counterfactual",
    generation_config={"n_cells": 500, "n_genes": 50, "n_celltype": 3, "n_concepts": 3},
    normalize=True,
    batch_size=32,
)

# 2. Generate the counterfactual dataset by setting concept_0 to 1
#    in cells of type 2.
intervened = dm.dataset.generate_interventions(
    {"concept_0": 1}, cell_type=2,
)

# 3. Compute the split from ORIGINAL concepts: exclude type 2/concept-on cells,
#    reserve half of type 2/concept-off cells for test, and train on the rest.
train_idxs, test_idxs = counterfactual_indices(
    dm.dataset, concept="concept_0", cell_type=2, seed=42,
)
dm.splitter = FixedIndicesSplitter(train_idxs=train_idxs, val_idxs=[], test_idxs=test_idxs)

# Modify the selected observed annotations in the original dataset according to the specified mode. Since only the original dataset is used for training, there is no need to modify the annotations in the intervened dataset.
if ANNOTATION_MODIFICATION["mode"] != "identity":
    observed = modify_annotations(dm.dataset.native_concepts, **ANNOTATION_MODIFICATION)
    dm.dataset.set_generated_concepts({"observed": observed}, use_as_gt=True)

# 4. Initialize the data loaders using the original dataset.
#    Training batches contain original (non-intervened) gene expression.
dm.setup()
train_batch = next(iter(dm.train_dataloader()))

# Report the number of cells assigned to each split and excluded
# from all splits.
excluded = len(dm.dataset) - sum((dm.splitter.train_len, dm.splitter.val_len, dm.splitter.test_len))
print(f"Train: {dm.splitter.train_len}; validation: {dm.splitter.val_len}; "
      f"test: {dm.splitter.test_len}; excluded: {excluded}")
print(f"Original training inputs: {tuple(train_batch['inputs']['x'].shape)}")

# 5. Retrieve paired original and counterfactual gene expression
#    profiles for the same test cells.
#    The counterfactual profiles serve as ground truth for evaluating
#    the model's predictions after intervention.
test_indices = dm.splitter.test_idxs
original_test = dm.dataset.input_data[test_indices]
counterfactual_test = intervened.input_data[test_indices]
print(f"Paired original/counterfactual test inputs: {tuple(original_test.shape)}")

# 6. Build a generative model: latent -> predicted concepts + unknown context
# -> gene expression. The encoder infers the latent from the original cell.
n_genes = dm.n_features[0]
n_concepts = len(dm.concept_names)
latent_size, embedding_size = 16, 8
model = ConceptBottleneckVAE(
    input_size=n_genes,
    annotations=dm.annotations,
    encoder=nn.Sequential(nn.Linear(n_genes, 64), nn.LeakyReLU(), nn.Linear(64, latent_size)),
    decoder=nn.Sequential(
        nn.Linear((n_concepts + 1) * embedding_size + n_concepts, 64),
        nn.LeakyReLU(), nn.Linear(64, n_genes),
    ),
    latent_size=latent_size,
    embedding_size=embedding_size,
    concepts_to_decoder=True,
    inference_kwargs={"p_int": 0.0},
    train_inference_kwargs={"p_int": 0.0},  # decode from predicted concepts during training
)
loss_fn = CompositeLoss(
    terms=[
        MSEReconstructionLoss(variable="input"),
        KLDivergenceLoss(latents=["z"]),
        ConceptLoss(binary=nn.BCEWithLogitsLoss()),
        OrthogonalityLoss("mixing", "unknown", n_concepts),
    ],
    weights=[1.0, 1e-4, 0.005, 0.05],
    names=["reconstruction", "kl", "concepts", "orthogonality"],
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# 7. Train on the original training cells with the selected observed annotations.
model.train()
for epoch in range(EPOCHS):
    total_loss, seen = 0.0, 0
    for batch in dm.train_dataloader():
        inputs = batch["inputs"]["x"]
        optimizer.zero_grad()
        out = model(query=model.prepare_query(batch), input=inputs)
        loss = loss_fn(out, {"x": inputs, "c": batch["concepts"]["c"]})
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
        seen += len(inputs)
    print(f"Epoch {epoch + 1}: loss = {total_loss / seen:.4f}")

# 8. Abduction: infer one latent per original test cell, then keep it fixed.
# The original CB_VAE.intervene also decodes from the posterior mean.
model.eval()
with torch.no_grad():
    posterior = model(query=list(model.pgm.variables), input=original_test)
    latent = posterior.guide_params["loc"]["z"].tensor

model.setup_inference(DeterministicInference)
query = ["input", "unknown", *dm.concept_names]
with torch.no_grad():
    reconstruction = model(query=query, evidence={"z": latent})

    # Action + prediction: force only concept_0 on; the other predicted concepts
    # and the unknown context still come from the SAME latent representation.
    # The concept head emits logits, so convert the desired probability to a logit.
    on_logit = torch.logit(torch.tensor(1.0), eps=1e-6)
    with intervention(model, DoIntervention(on_logit), UniformPolicy(), ["concept_0"]):
        counterfactual_prediction = model(query=query, evidence={"z": latent})

# 9. Compare model predictions. Calculate the MSE between the original reconstruction and the original test data, 
# and between the counterfactual prediction and the counterfactual test data.
original_prediction = reconstruction.value["input"].tensor
predicted_counterfactual = counterfactual_prediction.value["input"].tensor
mse = nn.functional.mse_loss
print("CF MSE, copy original cell:", mse(original_test, counterfactual_test).item())
print("CF MSE, model without intervention:", mse(original_prediction, counterfactual_test).item())
print("CF MSE, model with intervention:", mse(predicted_counterfactual, counterfactual_test).item())
