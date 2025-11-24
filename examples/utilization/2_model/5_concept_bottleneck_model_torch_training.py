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
from torch import nn
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss
from torch_concepts.data.datasets import ToyDataset
from torch.distributions import Bernoulli

from torchmetrics.classification import BinaryAccuracy


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 1000
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    concept_names = [dataset.concept_names[i] for i in concept_idx]
    task_names = [dataset.concept_names[i] for i in task_idx]
    
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

    # Initialize the CBM
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=concept_annotations,
        task_names=task_names,
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1}
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
    print("Step 4: Training loop with torch loss")
    print("=" * 60)

    n_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Concatenate concepts and tasks as target
        target = torch.cat([c_train, y_train], dim=1)

        # Forward pass - query all variables (concepts + tasks)
        logits = model(x_train, query=query)
        
        # Compute loss on all outputs
        loss = loss_fn(logits, target)
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)
    
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        logits = model(x_train, query=query)
        c_pred = logits[:, :n_concepts]
        y_pred = logits[:, n_concepts:]
        
        # Compute accuracy using BinaryAccuracy
        concept_acc = concept_acc_fn(c_pred, c_train.int()).item()
        task_acc = task_acc_fn(y_pred, y_train.int()).item()
        
        print(f"Concept accuracy: {concept_acc:.4f}")
        print(f"Task accuracy: {task_acc:.4f}")

if __name__ == "__main__":
    main()
