"""
Example: ConceptBottleneckModel with PyTorch Lightning Training

This example demonstrates how to initialize and train a ConceptBottleneckModel
using PyTorch Lightning.

The model uses:
- ConceptBottleneckModel
- lightning=True to enable Lightning training capabilities
- Different inference engines can be used for training vs evaluation
- Annotations for concept metadata
- Flexible loss functions and metrics
"""

import torch
from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, MLP
from torch_concepts.data import BnLearnDataModule

from torchmetrics.classification import BinaryAccuracy

from pytorch_lightning import Trainer

def main():

    seed_everything(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 10000
    batch_size = 512
    datamodule = BnLearnDataModule(name="asia", n_gen=n_samples, seed=42,
                                   val_size=0.1, test_size=0.2,
                                   batch_size=batch_size)
    annotations = datamodule.dataset.annotations
    task_names = ['dysp']
    concept_names = [n for n in annotations.labels if n not in task_names]
    query = concept_names + task_names
    n_concepts = len(concept_names)

    n_features = datamodule.dataset.n_features[-1]

    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Initialize the CBM
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
        # Enable Lightning training
        lightning=True,
        loss=torch.nn.BCEWithLogitsLoss(),
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.02}
    )
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Encoder output features: {model.latent_size}")

    # Test lightning training
    print("\n" + "=" * 60)
    print("Step 3: Training loop with lightning")
    print("=" * 60)

    trainer = Trainer(max_epochs=100)

    model.train()
    trainer.fit(model, datamodule=datamodule)

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation with standard torch metrics")
    print("=" * 60)
    
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    concept_acc_sum = 0.0
    task_acc_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        test_loader = datamodule.test_dataloader()
        for batch in test_loader:
            out = model(query=query, input=batch['inputs']['x'])
            c_pred = torch.cat([out.params[n]['logits'] for n in concept_names], dim=1)
            y_pred = torch.cat([out.params[n]['logits'] for n in task_names], dim=1)

            c = batch['concepts']['c']
            c_true = c[:, :n_concepts]
            y_true = c[:, n_concepts:]

            concept_acc_sum += concept_acc_fn(c_pred, c_true.int()).item()
            task_acc_sum += task_acc_fn(y_pred, y_true.int()).item()
            num_batches += 1

    avg_concept_acc = concept_acc_sum / num_batches if num_batches > 0 else 0.0
    avg_task_acc = task_acc_sum / num_batches if num_batches > 0 else 0.0

    print(f"Average concept accuracy: {avg_concept_acc:.4f}")
    print(f"Average task accuracy: {avg_task_acc:.4f}")

if __name__ == "__main__":
    main()