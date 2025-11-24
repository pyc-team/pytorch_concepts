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
from torch.utils.data import Dataset, DataLoader
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.data.datasets import ToyDataset
from torch.distributions import Bernoulli

from pytorch_lightning import Trainer

from torch_concepts.nn.modules.loss import ConceptLoss

class ConceptDataset(Dataset):
    """Custom dataset that returns batches in the format expected by ConceptBottleneckModel."""
    
    def __init__(self, x, c, y):
        self.x = x
        self.concepts = torch.cat([c, y], dim=1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {
            'inputs': {'x': self.x[idx]},
            'concepts': {'c': self.concepts[idx]},
        }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 1000
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train = data.data
    c_train = data.concept_labels 
    y_train = data.target_labels
    concept_names = data.concept_attr_names
    task_names = data.task_attr_names
    
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]
    
    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names}")
    print(f"Tasks: {n_tasks} - {task_names}")
    print(f"Training samples: {n_samples}")

    # For binary concepts, we can use simple labels
    concept_annotations = Annotations({
        1: AxisAnnotation(
            labels=tuple(concept_names + task_names),
            metadata={
                concept_names[0]: {
                    'type': 'discrete',
                    'distribution': Bernoulli
                },
                concept_names[1]: {
                    'type': 'discrete',
                    'distribution': Bernoulli
                },
                task_names[0]: {
                    'type': 'discrete',
                    'distribution': Bernoulli
                }
            }
        )
    })
    
    print(f"Concept axis labels: {concept_annotations[1].labels}")
    print(f"Concept types: {[concept_annotations[1].metadata[name]['type'] for name in concept_names]}")
    print(f"Concept cardinalities: {concept_annotations[1].cardinalities}")
    print(f"Concept distributions: {[concept_annotations[1].metadata[name]['distribution'] for name in concept_names]}")


    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Define loss function
    loss_fn = ConceptLoss(
        annotations=concept_annotations,
        fn_collection={
            'discrete': {
                'binary': {'path': "torch.nn.BCEWithLogitsLoss"}
                # all concept are discrete and binary in this example,
                # so we only need to define binary loss
            }
        }
    )

    # Define metrics
    metrics = {
        'discrete': {
            'binary': {
                'accuracy': {'path': "torchmetrics.classification.BinaryAccuracy"},
                'auc': {'path': "torchmetrics.classification.BinaryAUROC"}
            }
            # all concept are discrete and binary in this example,
            # so we only need to define binary metrics
        }
    }

    # Initialize the CBM
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=concept_annotations,
        task_names=task_names,
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        loss=loss_fn,
        metrics=metrics,
        summary_metrics=True,
        perconcept_metrics=True,
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
    
    batch_size = 8
    x_batch = x_train[:batch_size]
    
    # Forward pass
    query = list(concept_names) + list(task_names)
    print(f"Query variables: {query}")
    
    with torch.no_grad():
        logits = model(x_batch, query=query)
    
    print(f"Input shape: {x_batch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected output dim: {n_concepts + n_tasks}")


    # Test forward pass
    print("\n" + "=" * 60)
    print("Step 4: Training loop with lightning")
    print("=" * 60)

    trainer = Trainer(
        max_epochs=500,
        log_every_n_steps=10
    )

    # Create dataset and dataloader
    train_dataset = ConceptDataset(x_train, c_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    model.train()
    trainer.fit(model, train_dataloaders=train_dataloader)

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation with Internal Metrics")
    print("=" * 60)
    
    # The metrics are accumulated during training but reset at each epoch end by PyTorch Lightning
    # To see the final metrics, we need to manually evaluate on the data
    model.eval()
    model.train_metrics.reset()
    
    with torch.no_grad():
        # Run forward pass and re-accumulate metrics
        # these are automatically reset at each epoch end by PyTorch Lightning
        out = model(x_train, query=query)
        in_metric_dict = model.filter_output_for_metric(out, torch.cat([c_train, y_train], dim=1))
        model.update_metrics(in_metric_dict, model.train_metrics)
        
        # Compute accumulated metrics
        train_metrics = model.train_metrics.compute()
        
        print("\nInternal Training Metrics:")
        print("-" * 60)
        for metric_name, metric_value in train_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                print(f"{metric_name}: {metric_value.item():.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()
