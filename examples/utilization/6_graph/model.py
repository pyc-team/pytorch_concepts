"""
Example: Graph Generator Inside CausalCGM

The same components prepared on the data side in ``caching.py`` can instead
run inside the high-level model:

- ``GraphGeneratorLearnable('dagma_cgm')`` is a trainable model component,
  so its adjacency is optimized jointly with the CGM mechanisms.
- ``DAGMALoss`` penalizes cyclic learned structures.

This example does not precompute embeddings or a fixed graph. The graph is
randomly initialized and is then learned by gradient descent.

Flow:
1. Generate samples from the Asia Bayesian network and create the data splits.
2. Initialize a DAGMA-CGM graph generator without preparing data splits.
3. Build ``CausalCGM`` with the learnable graph generator.
4. Train all non-frozen components with a plain PyTorch loop.
5. Run one evaluation forward and inspect the graph materialized by CausalCGM.
"""
import torch

from torch_concepts import seed_everything
from torch_concepts.graph_generator import GraphGeneratorLearnable
from torch_concepts.graph_generator import remove_weakest_cycles, random_initialization
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn import CGMTrainingLoss, CausalCGM, MLP


GRAPH_THRESHOLD = 0.02
TASK = "dysp"
EPOCHS = 1
LIMIT_TRAIN_BATCHES = 3


def batch_xy(batch):
    return batch["inputs"]["x"], batch["concepts"]["c"]


def train_one_epoch(model, datamodule, loss_fn, optimizer, limit_batches=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(datamodule.train_dataloader()):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        inputs, target_concepts = batch_xy(batch)
        output = model(input=inputs, target=model.prepare_target(target_concepts))
        target = model.prepare_target(target_concepts, output)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_and_get_graph(model, datamodule):
    with torch.no_grad():
        batch = next(iter(datamodule.test_dataloader()))
        inputs, target = batch_xy(batch)
        model.eval()
        model(input=inputs, target=target)
        return model.graph


def main():
    seed_everything(42)
    dm = BnLearnDataModule(
        name="asia", n_gen=500, batch_size=64, seed=42,
    )
    # Random initialization needs no data split; Trainer.fit calls setup().
    graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=list(dm.annotations.labels),
        task_names=[TASK],
        threshold=GRAPH_THRESHOLD,
        no_out_task=True,
        refinement=remove_weakest_cycles,
        initialization=random_initialization,
    )
    model = CausalCGM(
        input_size=dm.n_features[-1],
        annotations=dm.annotations,
        task_names=TASK,
        backbone=MLP(dm.n_features[-1], 128),
        latent_size=128,
        embedding_size=8,
        graph_generator=graph_generator,
        lightning=False,
    )
    loss_fn = CGMTrainingLoss()
    loss_fn.configure_terms(model.graph_generator)
    if hasattr(loss_fn, "task_names"):
        loss_fn.task_names = list(model.task_names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    dm.setup("fit")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(
            model, dm, loss_fn, optimizer,
            limit_batches=LIMIT_TRAIN_BATCHES,
        )
        print(f"Epoch {epoch + 1}/{EPOCHS} loss: {loss:.4f}")

    # Evaluation materializes the learned graph inside CausalCGM.
    learned_graph = evaluate_and_get_graph(model, dm)
    print("Learned Asia graph:\n", learned_graph.to_pandas())


if __name__ == "__main__":
    main()
