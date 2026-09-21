"""Minimal reproduction of the CausalCGM dSprites demo.

Reference: https://github.com/gabriele-dominici/CausalCGM/blob/main/demo.ipynb
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import Annotations, seed_everything
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.nn import (
    CGMTrainingLoss,
    CausalCGM,
    ConceptLoss,
    ConceptMetrics,
    WeightedConceptLoss,
)
from torch_concepts.nn.functional import cace_score


SEED = 0
LABELS = ["Shape", "Size", "PosY", "PosX", "Color", "Label"]
TASK = "Label"

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


class EpochDiagnostics(Callback):
    """Print the epoch-level metrics accumulated by ``ConceptMetrics``."""

    names = (
        "train_loss", "val_loss",
        "train/SUMMARY-binary_accuracy",
        "val/SUMMARY-binary_accuracy",
    )

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        values = [
            f"{name}={float(metrics[name]):.3f}"
            for name in self.names if name in metrics
        ]
        print(f"Epoch {trainer.current_epoch:03d} | " + ", ".join(values))


def load_data(data_dir):
    """Load the train and test arrays provided by the original demo."""
    def load(name):
        path = data_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing dSprites array: {path}")
        value = torch.from_numpy(np.load(path)).float()
        return value[:, None] if value.ndim == 1 else value

    train_x = load("train_features")
    train_c = load("train_concepts")
    train_y = load("train_tasks")
    test_x = load("test_features")
    test_c = load("test_concepts")
    test_y = load("test_tasks")

    return (
        train_x,
        torch.cat([train_c, train_y], dim=1),
        test_x,
        torch.cat([test_c, test_y], dim=1),
    )


def make_datamodule(train_x, train_target, test_x, test_target, batch_size):
    """Build the standard datamodule with the exact demo split."""
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

    # Last 20% of the original train file is validation; test stays untouched.
    split = int(0.8 * len(train_x))
    splitter = FixedIndicesSplitter(
        train_idxs=range(split),
        val_idxs=range(split, len(train_x)),
        test_idxs=range(len(train_x), len(dataset)),
    )
    return ConceptDataModule(
        dataset=dataset,
        splitter=splitter,
        batch_size=batch_size,
        drop_last=False,
    )



def make_model(datamodule):
    """Create the fixed-graph model and the original training objective."""
    # The original demo averages concepts and task separately, then sums them.
    prediction_loss = WeightedConceptLoss(
        concept_weight=1.0,
        task_weight=1.0,
        task_names=[TASK],
        binary=torch.nn.BCEWithLogitsLoss(),
    )

    model = CausalCGM(
        input_size=datamodule.dataset.input_data.shape[1],
        annotations=datamodule.annotations,
        task_names=TASK,
        embedding_size=8,
        graph=datamodule.graph,
        inference_kwargs={"p_int": 1.0},
        lightning=True,
        loss=CGMTrainingLoss(
            prediction_loss=prediction_loss,
            lambda_dag=0.0,
            lambda_cace=0.05,
            evaluation_loss=ConceptLoss(
                binary=torch.nn.BCEWithLogitsLoss(),
            ),
        ),
        metrics=ConceptMetrics(
            datamodule.annotations,
            binary={"accuracy": BinaryAccuracy()},
        ),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
    )
    return model


def evaluate_interventions(model, inputs, targets):
    """Evaluate predictions and causal interventions directly through the model."""
    pos_x_index = LABELS.index("PosX")
    size_index = LABELS.index("Size")

    pos_x = targets[:, pos_x_index:pos_x_index + 1]
    size = targets[:, size_index:size_index + 1]

    low = torch.zeros_like(pos_x)
    high = torch.ones_like(pos_x)
    empty_query = dict.fromkeys(LABELS)

    model.eval()

    with torch.no_grad():
        # Ordinary observational prediction.
        prediction_output = model(input=inputs)

        # do(PosX = 0) and do(PosX = 1).
        do_low_output = model(
            input=inputs,
            query={**empty_query, "PosX": low},
        )
        do_high_output = model(
            input=inputs,
            query={**empty_query, "PosX": high},
        )

        # Block the path through Size by fixing it to its observed value.
        blocked_low_output = model(
            input=inputs,
            query={
                **empty_query,
                "PosX": low,
                "Size": size,
            },
        )
        blocked_high_output = model(
            input=inputs,
            query={
                **empty_query,
                "PosX": high,
                "Size": size,
            },
        )

    prediction = torch.cat(
        [
            prediction_output.params[name]["logits"].sigmoid()
            for name in LABELS
        ],
        dim=1,
    )

    do_low_label = do_low_output.params[TASK]["logits"].sigmoid()
    do_high_label = do_high_output.params[TASK]["logits"].sigmoid()

    blocked_low_label = (
        blocked_low_output.params[TASK]["logits"].sigmoid()
    )
    blocked_high_label = (
        blocked_high_output.params[TASK]["logits"].sigmoid()
    )

    accuracy = (
        (prediction > 0.5) == targets.bool()
    ).float().mean()

    cace = cace_score(
        do_low_label[:, -1],
        do_high_label[:, -1],
    ).abs().detach().item()

    cace_block = cace_score(
        blocked_low_label[:, -1],
        blocked_high_label[:, -1],
    ).abs().detach().item()

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"CACE PosX -> Label: {cace:.4f}")
    print(f"CACE after blocking Size: {cace_block:.4f}")

def main():
    seed_everything(SEED, workers=False)

    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data" / "dsprites_demo"
    batch_size = 128

    print("Step 1: load the dSprites arrays")
    train_x, train_target, test_x, test_target = load_data(data_dir)

    print("Step 2: create the dataset and datamodule")
    datamodule = make_datamodule(
        train_x, train_target, test_x, test_target, batch_size
    )

    print("Step 3: define the fixed causal graph and CausalCGM")
    model = make_model(datamodule)
    print(model.graph.data)

    print("Step 4: train the model")
    checkpoint = ModelCheckpoint(
        monitor="val/SUMMARY-binary_accuracy",
        mode="max",
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        max_epochs=200,
        accelerator="cpu",
        enable_checkpointing=True,
        callbacks=[checkpoint, EpochDiagnostics()],
    )
    trainer.fit(model, datamodule=datamodule)

    best = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(best["state_dict"])

    print("Step 5: evaluate predictions and causal interventions")
    evaluate_interventions(model, test_x, test_target)


if __name__ == "__main__":
    main()
