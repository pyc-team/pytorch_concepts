"""Reproduce the CausalCGM paper runner on dSprites with torch-concepts.

This is the dSprites-only counterpart of the authors' ``run.py``:
https://github.com/gabriele-dominici/CausalCGM/blob/main/run.py

It compares four paper-runner models over five seeds:

* ``CausalCGM`` learns a DAG, initialized from conditional entropy;
* ``CausalCGM_given`` uses the known dSprites DAG;
* ``CEM`` is the concept embedding baseline;
* ``CBM`` is the concept bottleneck baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.style as mpl_style

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import Annotations, seed_everything
from torch_concepts.graph_generator import (
    GraphGeneratorLearnable,
    entropy_initialization,
    fixed_dagma_initialization,
    remove_weakest_cycles,
)
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.nn import (
    CGMTrainingLoss,
    CausalCGM,
    ConceptBottleneckModel,
    ConceptEmbeddingModel,
    ConceptMetrics,
    WeightedConceptLoss,
)
from torch_concepts.nn.functional import cace_score


LABELS = ["Shape", "Size", "PosY", "PosX", "Color", "Label"]
TASK = "Label"
PERTURB = "PosX"
BLOCK = "Size"
N_SEEDS = 5
EPOCHS = 200
BATCH_SIZE = 128
# Match the released run.py and its CausalCGM/DAGMA defaults.
LAMBDA_DAG = 3.0
LAMBDA_CACE = 0.0
GRAPH_THRESHOLD = 0.02
OUTPUT_DIR = Path("outputs/16_example_cgm")

# adjacency[source, target] = 1 means source -> target
ADJACENCY = torch.tensor(
    [
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.float32,
)


def load_data(data_dir: Path):
    """Load the precomputed dSprites feature and label arrays.

    The file names match the arrays used by the original CausalCGM runner:
    ``*_features`` are frozen image embeddings, ``*_concepts`` are the five
    dSprites concepts, and ``*_tasks`` is the binary task label. Concepts and
    task are concatenated into the six-node order in :data:`LABELS`.
    """
    def load(name):
        path = data_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing dSprites array: {path}")
        value = torch.from_numpy(np.load(path)).float()
        return value[:, None] if value.ndim == 1 else value

    train_x = load("train_features")
    train_target = torch.cat(
        [load("train_concepts"), load("train_tasks")], dim=1,
    )
    test_x = load("test_features")
    test_target = torch.cat(
        [load("test_concepts"), load("test_tasks")], dim=1,
    )
    return train_x, train_target, test_x, test_target


def make_datamodule(train_x, train_target, test_x, test_target, batch_size):
    """Shuffle and split the original training set as in the paper runner.

    The original ``run.py`` shuffles the train arrays once per seed, then uses
    the first 80% for fitting and the last 20% for validation. The held-out test
    arrays remain untouched and are appended only so the datamodule can expose a
    standard test split.

    Returns
    -------
    tuple
        ``(datamodule, fit_target)`` where ``fit_target`` is read back from
        the datamodule train split and used to initialize the learned CGM graph
        with conditional entropy.
    """
    permutation = torch.randperm(len(train_x))
    train_x = train_x[permutation]
    train_target = train_target[permutation]
    annotations = Annotations(
        labels=LABELS,
        cardinalities=[1] * len(LABELS),
        types=["binary"] * len(LABELS),
    )
    dataset = ConceptDataset(
        input_data=torch.cat([train_x, test_x]),
        concepts=torch.cat([train_target, test_target]),
        annotations=annotations,
        graph=pd.DataFrame(ADJACENCY.numpy(), index=LABELS, columns=LABELS),
        name="dSprites",
    )
    split = int(0.8 * len(train_x))
    datamodule = ConceptDataModule(
        dataset=dataset,
        splitter=FixedIndicesSplitter(
            train_idxs=range(split),
            val_idxs=range(split, len(train_x)),
            test_idxs=range(len(train_x), len(dataset)),
        ),
        batch_size=batch_size,
    )
    datamodule.setup("fit")
    train_indices = datamodule.trainset.indices
    fit_target = datamodule.dataset.concepts.tensor[train_indices]
    return datamodule, fit_target


def make_models(datamodule, fit_target):
    """Create all paper-runner models for one seed and datamodule split.

    The learned CGM receives its initialized graph generator at construction
    time. The given-graph CGM mirrors the original runner by installing the
    known dSprites DAG as a frozen DAGMA-CGM ``fc1`` weight.
    """
    input_size = datamodule.dataset.input_data.shape[1]

    def metrics():
        return ConceptMetrics(
            datamodule.annotations, summary=True, per_concept=[TASK],
            binary={"accuracy": BinaryAccuracy()},
        )

    def backbone():
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 8),
        )

    learned_graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=LABELS,
        task_names=[TASK],
        threshold=GRAPH_THRESHOLD,
        refinement=remove_weakest_cycles,
        initialization=entropy_initialization(fit_target),
    )
    given_graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=LABELS,
        task_names=[TASK],
        threshold=GRAPH_THRESHOLD,
        refinement=remove_weakest_cycles,
        initialization=fixed_dagma_initialization(ADJACENCY),
    )
    def baseline_loss():
        return WeightedConceptLoss(
            concept_weight=1.0, task_weight=1.0,
            task_names=[TASK], binary=nn.BCEWithLogitsLoss(),
        )
    return {
        "CausalCGM": CausalCGM(
            input_size=input_size,
            annotations=datamodule.annotations,
            task_names=TASK,
            embedding_size=8,
            graph_generator=learned_graph_generator,
            run_interventions=True,
            lightning=True,
            loss=make_training_loss(),
            metrics=metrics(),
            optim_class=torch.optim.AdamW,
            optim_kwargs={"lr": 0.01},
        ),
        "CausalCGM_given": CausalCGM(
            input_size=input_size,
            annotations=datamodule.annotations,
            task_names=TASK,
            embedding_size=8,
            graph_generator=given_graph_generator,
            run_interventions=True,
            lightning=True,
            loss=make_training_loss(),
            metrics=metrics(),
            optim_class=torch.optim.AdamW,
            optim_kwargs={"lr": 0.01},
        ),
        "CEM": ConceptEmbeddingModel(
            input_size=input_size,
            annotations=datamodule.annotations,
            task_names=TASK,
            embedding_size=8,
            backbone=backbone(),
            latent_size=8,
            lightning=True,
            loss=baseline_loss(),
            metrics=metrics(),
            optim_class=torch.optim.AdamW,
            optim_kwargs={"lr": 0.01},
            plate=False,
        ),
        "CBM": ConceptBottleneckModel(
            input_size=input_size,
            annotations=datamodule.annotations,
            task_names=TASK,
            backbone=backbone(),
            latent_size=8,
            lightning=True,
            loss=baseline_loss(),
            metrics=metrics(),
            optim_class=torch.optim.AdamW,
            optim_kwargs={"lr": 0.01},
            plate=False,
        ),
    }


def make_training_loss(lambda_cace=LAMBDA_CACE):
    """Build the paper objective while keeping model outputs as logits."""
    prediction_loss = WeightedConceptLoss(
        concept_weight=1.0, task_weight=1.0, task_names=[TASK],
        binary=nn.BCEWithLogitsLoss(),
    )
    return CGMTrainingLoss(
        prediction_loss=prediction_loss,
        lambda_dag=LAMBDA_DAG,
        lambda_cace=lambda_cace,
    )


def interventions_from_root(model, inputs, target, adjacency):
    """Reproduce the paper's ``acc_int`` intervention curve.

    Nodes are ordered by number of reachable descendants in the materialized
    DAG. The function perturbs the inputs, then progressively intervenes on
    concepts in that order and records the change in accuracy over the nodes
    that remain un-intervened.
    """
    task_index = LABELS.index(TASK)
    dag = (adjacency > 0.1).float().cpu().numpy()
    graph = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Expected the materialized CGM graph to be a DAG.")

    # Rank nodes by number of reachable descendants, as in the paper utility.
    reachability = dag.copy()
    for source in range(dag.shape[0]):
        for target_index in range(dag.shape[1]):
            if source != target_index and nx.has_path(
                graph, source, target_index,
            ):
                reachability[source, target_index] = 1
    order = np.flip(np.argsort(reachability.sum(axis=1))).tolist()

    perturbed_inputs = inputs + torch.randn_like(inputs) * 15
    intervention_indices = []
    changes = []
    with torch.no_grad():
        baseline_output = model(
            input=perturbed_inputs,
            query={label: None for label in LABELS},
        )
        baseline = torch.cat([
            baseline_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)

    for current_node in order:
        # Equivalent to exclude=[] and exclude_labels=[task_index] in run.py.
        if current_node == task_index:
            continue
        intervention_indices.append(current_node)
        included = [
            index for index in range(len(LABELS))
            if index not in intervention_indices
        ]
        if not included:
            continue
        values = {
            LABELS[index]: target[:, index:index + 1]
            for index in intervention_indices
        }
        with torch.no_grad():
            intervened_output = model(
                input=perturbed_inputs,
                query={
                    label: None for label in LABELS
                    if label not in values
                },
                evidence=values,
            )
            intervened = torch.cat([
                values[label] if label in values
                else intervened_output.params[label]["logits"].sigmoid()
                for label in LABELS
            ], dim=1)
        baseline_accuracy = accuracy_score(
            target[:, included].cpu().numpy().ravel(),
            (baseline[:, included] > 0.5).cpu().numpy().ravel(),
        )
        intervention_accuracy = accuracy_score(
            target[:, included].cpu().numpy().ravel(),
            (intervened[:, included] > 0.5).cpu().numpy().ravel(),
        )
        changes.append(intervention_accuracy - baseline_accuracy)

    intervention_order = [
        LABELS[index] for index in order if index != task_index
    ]
    return changes, intervention_order


def plot_acc_int(intervention_rows):
    """Plot mean +/- standard error intervention curves as in the paper."""
    model_names = list(dict.fromkeys(row["model"] for row in intervention_rows))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, model_name in enumerate(model_names):
        color = colors[index % len(colors)]
        rows = [row for row in intervention_rows if row["model"] == model_name]
        curves = np.asarray([row["acc_int"] for row in rows], dtype=float) * 100
        curves = np.column_stack([np.zeros(len(curves)), curves])
        mean = curves.mean(axis=0)
        standard_error = curves.std(axis=0) / np.sqrt(len(curves))
        steps = np.arange(len(mean))

        axis.plot(
            steps, mean, marker="o", linewidth=2, color=color,
            label=model_name,
        )
        axis.fill_between(
            steps, mean - standard_error, mean + standard_error,
            color=color, alpha=0.2,
        )

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Number of intervened concepts")
    axis.set_xticks(steps)
    axis.set_ylabel(r"$\Delta$ label accuracy (percentage points)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title=r"mean $\pm$ standard error", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "acc_int.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_graph(adjacency, title, output_path):
    """Plot a thresholded adjacency heatmap, as in the paper Figure 14."""
    dag = (torch.as_tensor(adjacency).detach().cpu().numpy() > 0.01).astype(int)
    fig, axis = plt.subplots(figsize=(6, 5.5))
    image = axis.imshow(
        dag, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    fig.colorbar(image, ax=axis, ticks=np.linspace(0, 1, 6))
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_pns_matrix(model, inputs, dag):
    """Compute the paper lower and upper PNS bounds for reachable pairs.

    The DAG is first closed under reachability, as in the released utility, so
    indirect ancestors are included in the set of pairs whose PNS bounds are
    reported.
    """
    dag = dag.copy()
    graph = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    for source in range(dag.shape[0]):
        for target in range(dag.shape[1]):
            if source != target and nx.has_path(graph, source, target):
                dag[source, target] = 1
    lower = np.full((len(LABELS), len(LABELS)), np.nan)
    upper = np.full_like(lower, np.nan)

    with torch.no_grad():
        for source, source_name in enumerate(LABELS):
            zeros = torch.zeros(len(inputs), 1, device=inputs.device)
            ones = torch.ones_like(zeros)
            zero_evidence = {source_name: zeros}
            one_evidence = {source_name: ones}
            zero_output = model(
                input=inputs,
                query={
                    label: None for label in LABELS
                    if label not in zero_evidence
                },
                evidence=zero_evidence,
            )
            one_output = model(
                input=inputs,
                query={
                    label: None for label in LABELS
                    if label not in one_evidence
                },
                evidence=one_evidence,
            )
            predicted_zero = torch.cat([
                zero_evidence[label] if label in zero_evidence
                else zero_output.params[label]["logits"].sigmoid()
                for label in LABELS
            ], dim=1) > 0.5
            predicted_one = torch.cat([
                one_evidence[label] if label in one_evidence
                else one_output.params[label]["logits"].sigmoid()
                for label in LABELS
            ], dim=1) > 0.5
            probability_zero = predicted_zero.float().mean(dim=0).cpu().numpy()
            probability_one = predicted_one.float().mean(dim=0).cpu().numpy()

            for target, edge in enumerate(dag[source]):
                if edge != 0:
                    lower[source, target] = max(
                        0.0, probability_one[target] - probability_zero[target],
                    )
                    upper[source, target] = min(
                        probability_one[target], 1.0 - probability_zero[target],
                    )
    return np.round(lower, 2), np.round(upper, 2)


def pns_pair_matrix(lower, upper):
    """Serialize each PNS matrix entry as ``(lower, upper)``."""
    matrix = []
    for row in range(lower.shape[0]):
        values = []
        for column in range(lower.shape[1]):
            if np.isnan(lower[row, column]) or np.isnan(upper[row, column]):
                values.append(np.nan)
            else:
                values.append((lower[row, column], upper[row, column]))
        matrix.append(values)
    return matrix


def evaluate(model, inputs, target):
    """Compute observational accuracy and PosX/Size intervention metrics.

    ``accuracy`` is the paper-style mean over all six nodes, despite being
    written to CSV as ``label_accuracy`` for compatibility with the original
    result tables. ``task_accuracy`` is the accuracy of the final ``Label`` node
    alone.
    """
    perturb_index = LABELS.index(PERTURB)
    block_index = LABELS.index(BLOCK)
    low = torch.zeros(len(inputs), 1, device=inputs.device)
    high = torch.ones_like(low)
    observed_block = target[:, block_index:block_index + 1]

    with torch.no_grad():
        observed_output = model(
            input=inputs,
            query={label: None for label in LABELS},
        )
        materialized_graph = (
            model.graph.data
            if isinstance(model, CausalCGM)
            else ADJACENCY
        )
        observed = torch.cat([
            observed_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)
        low_evidence = {PERTURB: low}
        high_evidence = {PERTURB: high}
        do_low_output = model(
            input=inputs,
            query={
                label: None for label in LABELS
                if label not in low_evidence
            },
            evidence=low_evidence,
        )
        do_high_output = model(
            input=inputs,
            query={
                label: None for label in LABELS
                if label not in high_evidence
            },
            evidence=high_evidence,
        )
        do_low = torch.cat([
            low_evidence[label] if label in low_evidence
            else do_low_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)
        do_high = torch.cat([
            high_evidence[label] if label in high_evidence
            else do_high_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)
        blocked_low_evidence = {PERTURB: low, BLOCK: observed_block}
        blocked_high_evidence = {PERTURB: high, BLOCK: observed_block}
        blocked_low_output = model(
            input=inputs,
            query={
                label: None for label in LABELS
                if label not in blocked_low_evidence
            },
            evidence=blocked_low_evidence,
        )
        blocked_high_output = model(
            input=inputs,
            query={
                label: None for label in LABELS
                if label not in blocked_high_evidence
            },
            evidence=blocked_high_evidence,
        )
        blocked_low = torch.cat([
            blocked_low_evidence[label]
            if label in blocked_low_evidence
            else blocked_low_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)
        blocked_high = torch.cat([
            blocked_high_evidence[label]
            if label in blocked_high_evidence
            else blocked_high_output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)
    correct = ((observed > 0.5) == target.bool()).float()
    metrics = {
        "accuracy": correct.mean().item(),
        "concept_accuracy": correct[:, :-1].mean().item(),
        "task_accuracy": correct[:, -1].mean().item(),
        "cace": cace_score(do_low[:, -1], do_high[:, -1]).abs().item(),
        "cace_block": cace_score(
            blocked_low[:, -1], blocked_high[:, -1],
        ).abs().item(),
    }
    acc_int, intervention_order = interventions_from_root(
        model, inputs, target, materialized_graph,
    )
    return metrics, acc_int, intervention_order, materialized_graph





def main():
    """Train all paper baselines, evaluate them, and write tables/plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    train_x, train_target, test_x, test_target = load_data(
        root / "data" / "dsprites_demo"
    )
    accelerator = "cpu"  # run.py explicitly disables CUDA.
    rows = []
    intervention_rows = []
    all_pns_rows = []
    original_pns_rows = []
    (OUTPUT_DIR / "graphs").mkdir(exist_ok=True)
    plot_graph(
        ADJACENCY, "dSprites DAG", OUTPUT_DIR / "graphs" / "given_graph",
    )

    for seed in range(N_SEEDS):
        seed_everything(seed, workers=True)
        datamodule, fit_target = make_datamodule(
            train_x, train_target, test_x, test_target, BATCH_SIZE
        )
        models = make_models(datamodule, fit_target)

        for name, model in models.items():
            checkpoint = ModelCheckpoint(
                dirpath=OUTPUT_DIR / "checkpoints" / name / str(seed),
                monitor=(
                    "val/SUMMARY-binary_accuracy"
                ),
                mode="max",
                save_top_k=1, save_weights_only=True,
            )
            trainer = Trainer(
                max_epochs=EPOCHS, accelerator=accelerator, devices=1,
                logger=False, callbacks=[checkpoint], enable_progress_bar=True,
                enable_model_summary=False,
            )
            trainer.fit(model, datamodule=datamodule)
            state = torch.load(
                checkpoint.best_model_path, map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state["state_dict"])
            model.to("cpu").eval()
            metrics, acc_int, intervention_order, materialized_graph = evaluate(
                model, test_x, test_target,
            )
            pns_graph = materialized_graph.detach().cpu().numpy()
            if not isinstance(model, CausalCGM):
                pns_graph = np.zeros_like(pns_graph)
                pns_graph[:-1, -1] = 1
            pns_lower, pns_upper = compute_pns_matrix(
                model, test_x, pns_graph,
            )
            original_pns_rows.append({
                "dataset": "dsprites_dataset",
                "model": name,
                "seed": seed,
                "PNS": str(pns_pair_matrix(pns_lower, pns_upper)),
            })
            for source, target in zip(*np.where(~np.isnan(pns_lower))):
                all_pns_rows.append({
                    "model": name, "seed": seed,
                    "cause": LABELS[source], "effect": LABELS[target],
                    "pns_lower": pns_lower[source, target],
                    "pns_upper": pns_upper[source, target],
                })
            if name == "CausalCGM":
                plot_graph(
                    materialized_graph, "dSprites DAG",
                    OUTPUT_DIR / "graphs" / f"learned_graph_seed_{seed}",
                )
            rows.append({
                "model": name, "seed": seed,
                # Kept for paper-table compatibility: this is all-node accuracy.
                "label_accuracy": metrics["accuracy"],
                "concept_accuracy": metrics["concept_accuracy"],
                "task_accuracy": metrics["task_accuracy"],
                "cace": metrics["cace"],
                "cace_block": metrics["cace_block"],
            })
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "results_raw.csv", index=False)
            intervention_rows.append({
                "dataset": "dsprites_dataset", "model": name,
                "seed": seed, "acc_int": acc_int,
                "intervention_order": intervention_order,
            })

    results = pd.DataFrame(rows)
    pd.DataFrame(intervention_rows).to_csv(
        OUTPUT_DIR / "interventions.csv", index=False,
    )
    pns_results = pd.DataFrame(all_pns_rows)
    pns_results.to_csv(OUTPUT_DIR / "pns.csv", index=False)
    pns_originalformat(original_pns_rows).to_csv(
        OUTPUT_DIR / "pns_originalformat.csv", index=False,
    )
    plot_acc_int(intervention_rows)
    accuracy_summary = (
        results.groupby("model")["label_accuracy"].agg(["mean", "std"]) * 100
    )
    cace_summary = results.groupby("model")["cace"].agg(
        mean="mean", std="std", standard_error="sem",
    )
    cace_block_summary = results.groupby("model")["cace_block"].agg(
        mean="mean", std="std", standard_error="sem",
    )
    results["residual_cace"] = np.divide(
        results["cace_block"], results["cace"],
        out=np.zeros(len(results), dtype=float),
        where=results["cace"].to_numpy() != 0,
    ) * 100
    residual_cace = results.groupby("model")["residual_cace"]
    residual_cace_summary = pd.DataFrame({
        "mean": residual_cace.mean(),
        "standard_error": residual_cace.sem(),
    })
    accuracy_summary.to_csv(OUTPUT_DIR / "results_summary.csv")
    cace_summary.to_csv(OUTPUT_DIR / "cace_summary.csv")
    cace_block_summary.to_csv(OUTPUT_DIR / "cace_block_summary.csv")
    residual_cace_summary.to_csv(OUTPUT_DIR / "residual_cace_summary.csv")
    print("Label accuracy (%)")
    print(accuracy_summary.to_string())
    print("\nCaCE")
    print(cace_summary.to_string())
    print("\nBlocked CaCE")
    print(cace_block_summary.to_string())
    print("\nResidual CaCE (%)")
    print(residual_cace_summary.to_string())
    pns_upper = (
        pns_results.groupby(["model", "cause", "effect"])["pns_upper"]
        .mean().unstack("effect")
    )
    for model_name in pns_upper.index.get_level_values("model").unique():
        matrix = pns_upper.loc[model_name].reindex(
            index=LABELS, columns=LABELS,
        )
        print(f"\nPNS upper bound: {model_name}")
        print(matrix.to_string())


if __name__ == "__main__":
    main()
