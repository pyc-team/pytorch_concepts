"""
Example: Testing ConceptBottleneckModel_Joint Initialization

This example demonstrates how to initialize and test a ConceptBottleneckModel_Joint,
which is the high-level API for joint training of concepts and tasks.

The model uses:
- BipartiteModel as the underlying structure (concepts -> tasks)
- Joint training (concepts and tasks trained simultaneously)
- Annotations for concept metadata
- Flexible loss functions and metrics
"""

import torch
from torch.distributions import Bernoulli
from pytorch_lightning import Trainer
import torchmetrics

from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.utils import GroupConfig
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 10000
    batch_size = 2048
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    datamodule = ConceptDataModule(dataset=dataset, 
                                   batch_size=batch_size,
                                   val_size=0.1,
                                   test_size=0.2)
    annotations = dataset.annotations
    concept_names = annotations.get_axis_annotation(1).labels

    n_features = dataset.input_data.shape[1]
    n_concepts = 2
    n_tasks = 1

    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names[:2]}")
    print(f"Tasks: {n_tasks} - {concept_names[2]}")
    print(f"Training samples: {n_samples}")

    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Define loss function
    loss_fn = ConceptLoss(
        annotations = annotations,
        fn_collection = GroupConfig(
            binary = torch.nn.BCEWithLogitsLoss(),
            categorical = torch.nn.CrossEntropyLoss(),
            continuous = torch.nn.MSELoss()
        )
    )

    # Define variable distributions as Bernoulli
    variable_distributions = {name: Bernoulli for name in concept_names}
    
    metrics = ConceptMetrics(
        annotations = annotations,
        summary_metrics=True,
        perconcept_metrics=True,
        fn_collection = GroupConfig(
            binary = {'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
    )

    # Initialize the CBM
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        loss=loss_fn,
        metrics=metrics,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.02}
    )
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Encoder output features: {model.latent_size}")


    # Test forward pass
    print("\n" + "=" * 60)
    print("Step 3: Test forward pass")
    print("=" * 60)
    
    x_batch = dataset.input_data[:batch_size]
    
    # Forward pass
    query = concept_names
    print(f"Query variables: {query}")
    
    with torch.no_grad():
        endogenous = model(x_batch, query=query)
    
    print(f"Input shape: {x_batch.shape}")
    print(f"Output endogenous shape: {endogenous.shape}")
    print(f"Expected output dim: {n_concepts + n_tasks}")


    # Test lightning training
    print("\n" + "=" * 60)
    print("Step 4: Training loop with lightning")
    print("=" * 60)

    trainer = Trainer(max_epochs=100)

    model.train()
    trainer.fit(model, datamodule=datamodule)

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation with internally-stored metrics")
    print("=" * 60)
    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    main()
