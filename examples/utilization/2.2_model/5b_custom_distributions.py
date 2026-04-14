"""
Example: Custom Distributions and Activations in Annotations

This example demonstrates how to add custom distributions and activation
functions to annotations before passing them to the model.

By default, the model uses Bernoulli for binary concepts and Categorical
for categorical concepts. Here we override the defaults with
RelaxedBernoulli (with a temperature parameter) and a custom activation.

The two utility functions are:
- ``add_distribution_to_annotations``: add distribution classes per concept
- ``add_activation_to_annotations``: add activation functions per concept
"""

import torch
from functools import partial
from torch import nn
from torch.distributions import RelaxedBernoulli

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn.modules.utils import GroupConfig
from torch_concepts.utils import (
    add_distribution_to_annotations,
    add_activation_to_annotations,
)

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

    # Add custom distributions to annotations
    # Here we use RelaxedBernoulli instead of the default Bernoulli
    print("\n" + "=" * 60)
    print("Step 2: Add custom distributions and activations")
    print("=" * 60)

    all_names = concept_names + task_names

    # Custom distributions: RelaxedBernoulli for all concepts and tasks
    concept_annotations = add_distribution_to_annotations(
        annotations=concept_annotations, 
        distributions=GroupConfig(binary = (RelaxedBernoulli, {'temperature': 0.5}))
    )
    print(f"Added RelaxedBernoulli distribution for: {all_names}")

    # Custom activations: sigmoid for all concepts and tasks
    concept_annotations = add_activation_to_annotations(
        annotations=concept_annotations, 
        activations={name: torch.sigmoid for name in all_names}
    )
    print(f"Added sigmoid activation for: {all_names}")

    # Verify the annotations metadata
    axis_ann = concept_annotations.get_axis_annotation(1)
    for name in all_names:
        meta = axis_ann.metadata[name]
        print(f"  {name}: distribution={meta['distribution'].__name__}, "
              f"activation={meta['activation'].__name__}")

    # Init model
    print("\n" + "=" * 60)
    print("Step 3: Initialize ConceptBottleneckModel")
    print("=" * 60)

    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=concept_annotations,
        task_names=task_names,
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1}
    )

    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Encoder output features: {model.latent_size}")

    # Training loop
    print("\n" + "=" * 60)
    print("Step 4: Training loop")
    print("=" * 60)

    n_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    loss_fn = nn.BCEWithLogitsLoss()
    query = list(concept_names) + list(task_names)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        target = torch.cat([c_train, y_train], dim=1)
        out = model(x=x_train, query=query, return_logits=True)
        loss = loss_fn(out.logits, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        out = model(x=x_train, query=query)
        c_pred = out.probs[:, :n_concepts]
        y_pred = out.probs[:, n_concepts:]

        concept_acc = concept_acc_fn(c_pred, c_train.int()).item()
        task_acc = task_acc_fn(y_pred, y_train.int()).item()

        print(f"Concept accuracy: {concept_acc:.4f}")
        print(f"Task accuracy: {task_acc:.4f}")


if __name__ == "__main__":
    main()
