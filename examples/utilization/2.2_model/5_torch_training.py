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
from torch_concepts.nn import ConceptBottleneckModel, GraphConceptBottleneckModel, \
    CausallyReliableConceptBottleneckModel, MLP
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy



def main():

    seed_everything(42)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    annotations = dataset.annotations
    n_features = dataset.n_features[-1]

    task_names = ["dysp"]
    concept_names = [n for n in dataset.concept_names if n not in task_names]
    query = concept_names + task_names

    x_train = dataset.input_data
    c_train = dataset.concepts[concept_names]
    y_train = dataset.concepts[task_names]

    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Initialize the CBM (defaults for distributions and activations are handled internally)
    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
        latent_size=128,  # Output size of the backbone
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
    print(f"Output {concept_names[0]} shape: {out.params[concept_names[0]]['logits'].shape}")
    print(f"Output {concept_names[1]} shape: {out.params[concept_names[1]]['logits'].shape}")
    print(f"Output {task_names[0]} shape: {out.params[task_names[0]]['logits'].shape}")

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
        target = c_train.union_with(y_train).float()
        
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
