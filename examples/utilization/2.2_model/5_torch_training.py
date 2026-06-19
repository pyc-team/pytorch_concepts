"""
Example: ConceptBottleneckModel with Manual PyTorch Training

This example demonstrates how to initialize and train a ConceptBottleneckModel
using a manual PyTorch training loop (without Lightning).

The model uses:
- ConceptBottleneckModel
- lightning=False (default) for pure PyTorch module behavior
- Manual optimizer and loss function setup
- Annotations for concept metadata
"""

import torch
from torch import nn

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, MLP
from torch_concepts.data import ToyDataset

from torchmetrics.classification import BinaryAccuracy



def main():

    seed_everything(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 1000
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    c_train = dataset.concepts[:, :2]
    y_train = dataset.concepts[:, 2:]
    concept_names = dataset.concept_names[:2]
    task_names = dataset.concept_names[2:]
    
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]
    
    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names}")
    print(f"Tasks: {n_tasks} - {task_names}")
    print(f"Training samples: {n_samples}")

    concept_annotations = dataset.annotations
    
    print(f"Concept axis labels: {concept_annotations[1].labels}")
    print(f"Concept types: {[concept_annotations[1].metadata[name]['type'] for name in concept_names]}")
    print(f"Concept cardinalities: {concept_annotations[1].cardinalities}")

    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Initialize the CBM (defaults for distributions and activations are handled internally)
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=concept_annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,  # Output size of the backbone
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
        out = model(query=query, input=x_batch)
    
    print(f"Input shape: {x_batch.shape}")
    print(f"Output C1 shape: {out.params['C1']['logits'].shape}")
    print(f"Output C2 shape: {out.params['C2']['logits'].shape}")
    print(f"Output xor shape: {out.params['xor']['logits'].shape}")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Step 4: Training loop with torch loss")
    print("=" * 60)

    n_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Concatenate concepts and tasks as target
        target = torch.cat([c_train, y_train], dim=1)

        # Forward pass - query all variables (concepts + tasks)
        out = model(query=query, input=x_train)
        
        # Compute loss on all outputs
        logits = torch.cat([out.params[name]['logits'] for name in query], dim=1)
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
        out = model(query=query, input=x_train)
        c_pred = torch.cat([out.params[name]['logits'] for name in concept_names], dim=1)
        y_pred = torch.cat([out.params[name]['logits'] for name in task_names], dim=1)
        
        # Compute accuracy using BinaryAccuracy
        concept_acc = concept_acc_fn(c_pred, c_train.int()).item()
        task_acc = task_acc_fn(y_pred, y_train.int()).item()
        
        print(f"Concept accuracy: {concept_acc:.4f}")
        print(f"Task accuracy: {task_acc:.4f}")

if __name__ == "__main__":
    main()
