"""Compare fixed-graph and learned-graph CGMs with mixed concept types.

Run from the repository root:
    python examples/utilization/2_model/17_example_cgm_mixed_variables.py
Quick end-to-end check:
    python examples/utilization/2_model/17_example_cgm_mixed_variables.py --epochs 1 --samples 256

Targets contain four columns: binary, category INDEX (0/1/2), continuous,
and continuous task. Output parameters instead have four logits (1 + 3)
and two locations/scales. MSE supervises locations, not Normal scales.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError

from torch_concepts import Annotations
from torch_concepts.graph_generator import (
    GraphGeneratorLearnable,
    fixed_dagma_initialization,
    random_initialization,
    remove_weakest_cycles,
)
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.nn import (
    CausalCGM, CGMTrainingLoss, ConceptLoss, ConceptMetrics, MLP,
)


LABELS = ["binary", "category", "continuous", "task"]
TASK = "task"
LAMBDA_DAG = 3.0
LAMBDA_CACE = 0.0
GRAPH_THRESHOLD = 0.02
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "17_example_cgm_mixed_variables"
# Rows are sources, columns are targets: binary -> continuous,
# category -> continuous, category -> task, continuous -> task.
ADJACENCY = torch.tensor([
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=torch.float32)


def make_datamodule(n_samples, batch_size):
    """Generate an SCM and noisy input measurements without task leakage."""
    binary = torch.randint(2, (n_samples,)).float()
    category = torch.randint(3, (n_samples,))
    category_effect = torch.tensor([-1.0, 0.0, 1.0])[category]
    continuous = (
        0.8 * (2 * binary - 1) + 0.7 * category_effect
        + 0.25 * torch.randn(n_samples)
    )
    task = (
        0.8 * continuous + 0.3 * torch.sin(continuous)
        + 0.4 * category_effect + 0.15 * torch.randn(n_samples)
    )
    targets = torch.stack([binary, category.float(), continuous, task], dim=1)

    # Measurements of the three concepts, mixed into 12 observed features.
    # The task and its independent noise are never used to construct x.
    signals = torch.cat([
        (2 * binary - 1)[:, None],
        torch.nn.functional.one_hot(category, num_classes=3).float(),
        continuous[:, None],
    ], dim=1)
    projection = torch.randn(5, 12) / (5 ** 0.5)
    inputs = signals @ projection + 0.15 * torch.randn(n_samples, 12)
    annotations = Annotations(
        labels=LABELS, cardinalities=[1, 3, 1, 1],
        types=["binary", "categorical", "continuous", "continuous"],
    )
    dataset = ConceptDataset(
        input_data=inputs, concepts=targets, annotations=annotations,
        graph=pd.DataFrame(ADJACENCY.numpy(), index=LABELS, columns=LABELS),
        name="mixed_cgm_toy",
    )
    dm = ConceptDataModule(
        dataset=dataset, batch_size=batch_size, val_size=0.1, test_size=0.2,
        workers=0,
    )
    dm.setup("fit")
    return dm


def make_training_loss(lambda_cace=LAMBDA_CACE):
    return CGMTrainingLoss(
        prediction_loss=ConceptLoss(
            binary=torch.nn.BCEWithLogitsLoss(),
            categorical=torch.nn.CrossEntropyLoss(),
            continuous=torch.nn.MSELoss(),
        ),
        lambda_dag=LAMBDA_DAG,
        lambda_cace=lambda_cace,
    )


def make_metrics(annotations):
    return ConceptMetrics(
        annotations=annotations,
        summary=False,
        per_concept=True,
        binary={"accuracy": BinaryAccuracy()},
        categorical={"accuracy": MulticlassAccuracy(num_classes=3)},
        continuous={"mse": MeanSquaredError()},
    )


def make_models(dm):
    learned_graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=LABELS,
        task_names=[TASK],
        threshold=GRAPH_THRESHOLD,
        no_out_task=True,
        refinement=remove_weakest_cycles,
        initialization=random_initialization,
    )
    given_graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=LABELS,
        task_names=[TASK],
        threshold=GRAPH_THRESHOLD,
        no_out_task=True,
        refinement=remove_weakest_cycles,
        initialization=fixed_dagma_initialization(ADJACENCY),
    )

    def build(graph_generator):
        return CausalCGM(
            input_size=12,
            annotations=dm.annotations,
            task_names=[TASK],
            embedding_size=16,
            backbone=MLP(input_size=12, hidden_size=64, n_layers=1),
            latent_size=64,
            lightning=True,
            # Mixed variables currently have no common random intervention grid.
            run_interventions=False,
            loss=make_training_loss(),
            metrics=make_metrics(dm.annotations),
            optim_class=torch.optim.AdamW,
            optim_kwargs={"lr": 0.001},
            graph_generator=graph_generator,
        )

    return {
        "CGM_given": build(given_graph_generator),
        "CGM_learned": build(learned_graph_generator),
    }


def check_training_forward(model, inputs, targets):
    with torch.no_grad():
        output = model(input=inputs, target=model.prepare_target(targets))
    for quantity in ("logits", "loc", "scale", "prior_logits", "prior_loc"):
        print(f"  {quantity}: {tuple(output.params[quantity].shape)}")
    model.graph_layer.clear()


def save_graph(name, matrix):
    pd.DataFrame(matrix.numpy(), index=LABELS, columns=LABELS).to_csv(
        OUTPUT_DIR / f"{name}_adjacency.csv"
    )


def plot_graphs(graphs):
    figure, axes = plt.subplots(1, len(graphs), figsize=(13, 4))
    if len(graphs) == 1:
        axes = [axes]
    vmax = max(1.0, max(float(matrix.max()) for matrix in graphs.values()))
    for axis, (name, matrix) in zip(axes, graphs.items()):
        axis.imshow(matrix.numpy(), vmin=0, vmax=vmax, cmap="Blues")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set(title=name, xlabel="Target", ylabel="Source")
        for source in range(len(LABELS)):
            for target in range(len(LABELS)):
                axis.text(
                    target,
                    source,
                    f"{matrix[source, target]:.2f}",
                    ha="center",
                    va="center",
                )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "graphs.png", dpi=160)
    plt.close(figure)


def evaluate(model, datamodule):
    with torch.no_grad():
        batch = next(iter(datamodule.test_dataloader()))
        inputs, concepts, transforms = model.unpack_batch(batch)
        inputs = model.maybe_scale_inputs(inputs, transforms)
        scaled_concepts = model.maybe_scale_concepts(concepts, transforms)
        c_loss = scaled_concepts.get("c", None)
        query = model.default_query(c_loss, "test")
        evidence = model.default_evidence(inputs, "test")
        model.eval()
        output = model(query=query, evidence=evidence)
        prepared_target = model.prepare_target(c_loss, output)
        losses = model.loss.breakdown(output, prepared_target)
        row = {
            f"eval_{name}_loss": float(value.detach().cpu())
            for name, value in losses.items()
        }
        row["eval_loss"] = sum(row.values())
        metric_output = model.unscale_output(output, transforms)
        metric_target = model.prepare_target(concepts.get("c", None), output)
        model.test_metrics.reset()
        model.test_metrics.update(metric_output, metric_target)
        metric_values = model.test_metrics.compute()
        model.test_metrics.reset()
        row.update({
            f"eval/{name}": float(value.detach().cpu())
            for name, value in metric_values.items()
        })
        graph = model.graph_layer.adjacency.detach().cpu()
        return row, graph


def train_and_evaluate(name, model, dm, epochs, batch_size):
    print(f"\nStep 2: Initialize {name}")

    print("Step 3: Check training forward pass")
    inputs = dm.dataset.input_data[:batch_size]
    targets = dm.dataset.concepts[:batch_size]
    check_training_forward(model, inputs, targets)

    print("Step 4: Train with Lightning")
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(OUTPUT_DIR / name),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

    print("Step 5: Evaluate with one forward on the held-out test split")
    metrics, graph = evaluate(model, dm)
    save_graph(name, graph)
    return {"model": name, **metrics}, graph


def main(epochs=100, n_samples=5000, batch_size=256):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 1: Generate mixed synthetic data")
    dm = make_datamodule(n_samples, batch_size)
    print(f"Inputs: {tuple(dm.dataset.input_data.shape)}")
    print(f"Targets: {tuple(dm.dataset.concepts.shape)}; columns: {LABELS}")
    print("Category targets are indices 0/1/2, not one-hot vectors.")
    models = make_models(dm)
    results = []
    graphs = {"Ground truth": ADJACENCY}

    for name, model in models.items():
        row, graph = train_and_evaluate(
            name=name,
            model=model,
            dm=dm,
            epochs=epochs,
            batch_size=batch_size,
        )
        results.append(row)
        graphs[name] = graph

    print("\nStep 6: Save metrics and compare materialized DAGs")
    table = pd.DataFrame(results).set_index("model")
    table = table[[
        column for column in table.columns
        if column.startswith("eval/test/")
    ]]
    table = table.rename(columns={
        "eval/test/continuous_mse": "eval/test/continuous_concept_mse",
        "eval/test/task_mse": "eval/test/continuous_task_mse",
    })
    table.to_csv(OUTPUT_DIR / "metrics.csv")
    print(table.to_string())
    plot_graphs(graphs)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 32 or args.batch_size < 1:
        parser.error("Require epochs >= 1, samples >= 32 and batch-size >= 1.")
    main(args.epochs, args.samples, args.batch_size)
