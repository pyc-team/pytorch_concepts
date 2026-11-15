"""
Swiss-Prot Concept Bottleneck Model
==================================

Create a disk-backed dataset and DataLoaders, then train a small CBM on CPU.
The default reads all Swiss-Prot records and the first 1,000,000 UniRef50
members with UniProtKB accessions, retaining only Swiss-Prot records matched
within that prefix. Use --max-uniref-members to change the limit, --full for
the complete UniRef50 membership, or --swissprot-only to skip UniRef50.

Inputs are 20 amino-acid frequencies and normalized sequence length. The
bottleneck predicts aromaticity, isoelectric point, hydropathy and helix
fraction; the task head predicts molecular weight in its original units.
This is a sequence-composition regression demo.

Examples:
    python examples/data/09_swissprot.py
    python examples/data/09_swissprot.py --max-uniref-members 500000
    python examples/data/09_swissprot.py --full
    python examples/data/09_swissprot.py --swissprot-only --workers 2
    python examples/data/09_swissprot.py --workers 4

Raw downloads and the SQLite index live in data/uniref_swissprot. The first
build keeps distinct Swiss-Prot accessions even when their sequences are
identical, merges UniRef50 and removes unmatched records before calculating
properties. It prints the retained count against the initial Swiss-Prot records.
The final dataset count is also printed after training and evaluation.
Allow time and disk space. Following runs reuse that cache.
Requires biopython for protein properties (pip install biopython).
"""

import argparse
from pathlib import Path

import torch

from torch_concepts import seed_everything
from torch_concepts.data.datamodules.uniref_swissprot import UniRefSwissProtDataModule
from torch_concepts.nn import ConceptBottleneckModel, WeightedConceptLoss

MAX_LENGTH = 512


def sequence_features(sequences):
    """Encode raw strings using sequence composition and length."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return torch.tensor([
        [sequence.count(aa) / len(sequence) for aa in amino_acids]
        + [len(sequence) / MAX_LENGTH]
        for sequence in sequences
    ], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Read all UniRef50 members.")
    parser.add_argument("--max-uniref-members", type=int, default=1_000_000,
                        help="Maximum UniRef50 members with accessions to read (default: 1000000).")
    parser.add_argument("--swissprot-only", action="store_true", help="Exclude UniRef50.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    seed_everything(42)

    # Dataset and DataLoaders.
    concept_names = ["aromaticity", "isoelectric_point", "gravy", "helix_fraction"]
    task_names = ["molecular_weight"]
    dm = UniRefSwissProtDataModule(
        root=Path(__file__).resolve().parents[2] / "data" / "uniref_swissprot",
        include_uniref=not args.swissprot_only,
        max_sequences=None,
        max_uniref_members=None if args.full else args.max_uniref_members,
        max_length=MAX_LENGTH,
        concept_subset=concept_names + task_names,
        val_size=0.1,
        test_size=0.2,
        batch_size=64,
        workers=args.workers,
        seed=42,
    )
    dataset = dm.dataset
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    print(f"Dataset: {len(dataset)} proteins, {dataset.n_concepts} selected labels")
    print(f"Train: {dm.train_len}, validation: {dm.val_len}, test: {dm.test_len}")

    # The task head sees only the predicted concepts.
    model = ConceptBottleneckModel(
        input_size=21,
        annotations=dataset.annotations,
        task_names=task_names,
        backbone=torch.nn.Sequential(torch.nn.Linear(21, 32), torch.nn.ReLU()),
        latent_size=32,
    )
    loss_fn = WeightedConceptLoss(
        concept_weight=1.0,
        task_weight=1.0,
        task_names=task_names,
        continuous=torch.nn.MSELoss(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            inputs = sequence_features(batch["inputs"]["x"])
            optimizer.zero_grad()
            out = model(query=dataset.concept_names, input=inputs)
            loss = loss_fn(out, batch["concepts"]["c"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            seen += len(inputs)
        print(f"Epoch {epoch + 1}: loss = {total_loss / seen:.4f}")

    # Continuous predictions are the Normal distributions' means (loc).
    model.eval()
    squared_error, seen = torch.zeros(dataset.n_concepts), 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                query=dataset.concept_names,
                input=sequence_features(batch["inputs"]["x"]),
            )
            predictions = out.loc[dataset.concept_names].tensor
            targets = batch["concepts"]["c"].tensor
            squared_error += (predictions - targets).square().sum(dim=0)
            seen += len(targets)
    for name, mse in zip(dataset.concept_names, squared_error / seen):
        print(f"Test MSE of {name}: {mse.item():.4f}")
    print(f"Final dataset: {len(dataset):,} Swiss-Prot records.")


if __name__ == "__main__":
    main()
