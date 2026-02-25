"""
Example: Comparing Different Training Modes

This example demonstrates how to train a ConceptBottleneckModel with different
training modes: joint and independent.

Training modes:
- 'joint': Train all concepts and tasks simultaneously (standard CBM)
- 'independent': Train level-by-level with ground truth from previous levels
"""

import torch
from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch.distributions import Bernoulli

from torchmetrics.classification import BinaryAccuracy

from pytorch_lightning import Trainer


def evaluate(model, datamodule, n_concepts, query):
    """Evaluate model on test set and return concept/task accuracy."""
    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    concept_acc_sum = 0.0
    task_acc_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        test_loader = datamodule.test_dataloader()
        for batch in test_loader:
            endogenous = model(x=batch['inputs']['x'], query=query)
            c_pred = endogenous[:, :n_concepts]
            y_pred = endogenous[:, n_concepts:]

            c_true = batch['concepts']['c'][:, :n_concepts]
            y_true = batch['concepts']['c'][:, n_concepts:]

            concept_acc = concept_acc_fn(c_pred, c_true.int()).item()
            task_acc = task_acc_fn(y_pred, y_true.int()).item()

            concept_acc_sum += concept_acc
            task_acc_sum += task_acc
            num_batches += 1

    avg_concept_acc = concept_acc_sum / num_batches if num_batches > 0 else 0.0
    avg_task_acc = task_acc_sum / num_batches if num_batches > 0 else 0.0

    print(f"Concept accuracy: {avg_concept_acc:.4f}")
    print(f"Task accuracy: {avg_task_acc:.4f}")
    
    return avg_concept_acc, avg_task_acc


def main():
    seed = 42
    seed_everything(seed)
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate toy XOR dataset")
    print("=" * 60)
    
    n_samples = 10000
    batch_size = 2048
    dataset = ToyDataset(dataset='xor', seed=seed, n_gen=n_samples)
    datamodule = ConceptDataModule(dataset=dataset, 
                                   batch_size=batch_size,
                                   val_size=0.1,
                                   test_size=0.2)
    datamodule.setup()
    annotations = dataset.annotations
    concept_names = annotations.get_axis_annotation(1).labels

    n_features = dataset.input_data.shape[1]
    n_concepts = 2
    query = concept_names

    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names[:2]}")
    print(f"Tasks: 1 - {concept_names[2]}")
    print(f"Training samples: {n_samples}")

    # Define variable distributions as Bernoulli
    variable_distributions = {name: Bernoulli for name in concept_names}
    loss = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW
    optim_kwargs = {'lr': 0.1}

    # =========================================================================
    # JOINT TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training mode: JOINT")
    print("=" * 60)

    model_joint = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        # lightning kwargs
        training='joint',
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_joint).__name__}")

    trainer_joint = Trainer(max_epochs=100)
    trainer_joint.fit(model_joint, datamodule=datamodule)
    evaluate(model_joint, datamodule, n_concepts, query)

    # =========================================================================
    # INDEPENDENT TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training mode: INDEPENDENT")
    print("=" * 60)

    model_independent = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        # lightning kwargs
        training='independent',
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_independent).__name__}")

    trainer_independent = Trainer(max_epochs=100)
    trainer_independent.fit(model_independent, datamodule=datamodule)
    evaluate(model_independent, datamodule, n_concepts, query)


if __name__ == "__main__":
    main()