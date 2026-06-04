"""
Example: Using Different Inference Engines for Training vs Evaluation

This example demonstrates how to train a ConceptBottleneckModel with 
different inference engines during training and evaluation.

Key concepts:
- `eval_inference`: Used during validation/testing (model.eval())
- `train_inference`: Used during training (model.train())

The active inference engine is selected automatically via PyTorch's
built-in train/eval mode.  Calling `model.train()` activates
`train_inference`; calling `model.eval()` activates
`eval_inference`.  Lightning toggles this automatically.

Current inference options:
- DeterministicInference: Returns logits directly (standard behavior)
- AncestralSamplingInference: Samples from distributions

Note: Independent training (where each level uses ground truth from previous levels)
can be implemented by creating a custom train_inference that uses evidence.
"""

import torch
from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, ConceptEmbeddingModel
from torch_concepts.nn.modules.mid.inference import (
    DeterministicInference,
    IndependentInference
)
from torch_concepts.data import ToyDataset
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
            # model.eval() automatically selects eval_inference
            out = model(x=batch['inputs']['x'], query=query)
            c_pred = out.probs[:, :n_concepts]
            y_pred = out.probs[:, n_concepts:]

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
                                   test_size=0.2,
                                   seed=seed)
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
    # STANDARD TRAINING (same inference for train and eval)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Standard Training (DeterministicInference)")
    print("=" * 60)

    model_standard = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        # Inference engines (both default to DeterministicInference)
        inference=DeterministicInference,
        train_inference=DeterministicInference,
        # Lightning kwargs
        lightning=True,
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_standard).__name__}")
    print(f"Inference (eval): {model_standard.eval_inference.__class__.__name__}")
    print(f"Training inference: {model_standard.train_inference.__class__.__name__}")

    trainer_standard = Trainer(max_epochs=100)
    trainer_standard.fit(model_standard, datamodule=datamodule)
    evaluate(model_standard, datamodule, n_concepts, query)

    # =========================================================================
    # DIFFERENT TRAINING MODE: INDEPENDENT TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Independent Training with Different Inference Engines")
    print("=" * 60)
    print("Uses IndependentInference for training")
    print("Uses DeterministicInference for evaluation")

    model_sampling = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        # Different inference for train vs eval
        inference=DeterministicInference,        # Eval: deterministic
        train_inference=IndependentInference, # Train: independent (uses GT concepts as evidence)
        # Lightning kwargs
        lightning=True,
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_sampling).__name__}")
    print(f"Eval inference: {model_sampling.eval_inference.__class__.__name__}")
    print(f"Training inference: {model_sampling.train_inference.__class__.__name__}")

    trainer_sampling = Trainer(max_epochs=100)
    trainer_sampling.fit(model_sampling, datamodule=datamodule)
    evaluate(model_sampling, datamodule, n_concepts, query)

    # =========================================================================
    # CEM WITH INDEPENDENT TRAINING (handles exogenous variables)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: CEM with Independent Training")
    print("=" * 60)
    print("Tests exogenous variable handling in IndependentInference")

    model_cem = ConceptEmbeddingModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        embedding_size=4,
        latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 1},
        # Different inference for train vs eval
        inference=DeterministicInference,        # Eval: deterministic
        train_inference=IndependentInference, # Train: independent (uses GT concepts as evidence)
        lightning=True,
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_cem).__name__}")
    print(f"Eval inference: {model_cem.eval_inference.__class__.__name__}")
    print(f"Training inference: {model_cem.train_inference.__class__.__name__}")

    trainer_cem = Trainer(max_epochs=100)
    trainer_cem.fit(model_cem, datamodule=datamodule)
    evaluate(model_cem, datamodule, n_concepts, query)


if __name__ == "__main__":
    main()