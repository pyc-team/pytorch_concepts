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

Train and eval must use the *same* inference class; the two regimes are
differentiated through ``train_inference_kwargs``.  Independent training (each
level conditioned on ground-truth parents) is obtained by teacher-forcing the
ground-truth concepts during training only, via ``p_int=1.0`` on the training
engine, while evaluation keeps the default ``p_int=0.0``.
"""

import torch
from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, ConceptEmbeddingModel, MLP, ConceptMemoryReasoner, CMRBlendedLoss, ConceptLoss
from torch_concepts.nn import DeterministicInference, IndependentInference
from torch_concepts.data import ToyDataset
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch.distributions import Bernoulli

from torchmetrics.classification import BinaryAccuracy

from pytorch_lightning import Trainer


def evaluate(model, datamodule, n_concepts, query):
    """Evaluate model on a data split and return concept/task accuracy."""
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
            out = model(input=batch['inputs']['x'], query=query)
            predictions = out.logits if out.logits is not None else out.probs
            c_pred = predictions[:, :n_concepts]
            y_pred = predictions[:, n_concepts:]
            if out.logits is not None:
                c_pred = torch.sigmoid(c_pred)
                y_pred = torch.sigmoid(y_pred)

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
    concept_names = annotations.labels

    n_features = dataset.input_data.shape[1]
    n_concepts = 2
    query = concept_names

    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names[:2]}")
    print(f"Tasks: 1 - {concept_names[2]}")
    print(f"Training samples: {n_samples}")

    # Define variable distributions as Bernoulli
    variable_distributions = {name: Bernoulli for name in concept_names}
    loss = ConceptLoss(binary=torch.nn.BCEWithLogitsLoss(), binary_param="logits")
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
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,
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
    print("Example 2: Independent Training via teacher forcing (p_int)")
    print("=" * 60)
    print("Train: DeterministicInference with p_int=1.0 (teacher-force GT concepts)")
    print("Eval:  DeterministicInference with p_int=0.0 (no teacher forcing)")

    model_sampling = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,
        inference=DeterministicInference,
        train_inference=IndependentInference,
        # Lightning kwargs
        lightning=True,
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_sampling).__name__}")
    print(f"Eval inference: {model_sampling.eval_inference.__class__.__name__} (p_int={model_sampling.eval_inference.p_int})")
    print(f"Training inference: {model_sampling.train_inference.__class__.__name__} (p_int={model_sampling.train_inference.p_int})")

    trainer_sampling = Trainer(max_epochs=100)
    trainer_sampling.fit(model_sampling, datamodule=datamodule)
    evaluate(model_sampling, datamodule, n_concepts, query)

    # =========================================================================
    # CEM WITH INDEPENDENT TRAINING (handles exogenous variables)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: CEM with Independent Training (teacher forcing)")
    print("=" * 60)
    print("Tests exogenous variable handling with p_int=1.0 teacher forcing")

    model_cem = ConceptEmbeddingModel(
        input_size=n_features,
        annotations=annotations,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        embedding_size=4,
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,
        inference=DeterministicInference,
        train_inference=IndependentInference,
        lightning=True,
        loss=loss,
        optim_class=optim,
        optim_kwargs=optim_kwargs
    )
    print(f"Model type: {type(model_cem).__name__}")
    print(f"Eval inference: {model_cem.eval_inference.__class__.__name__} (p_int={model_cem.eval_inference.p_int})")
    print(f"Training inference: {model_cem.train_inference.__class__.__name__} (p_int={model_cem.train_inference.p_int})")

    trainer_cem = Trainer(max_epochs=100)
    trainer_cem.fit(model_cem, datamodule=datamodule)
    evaluate(model_cem, datamodule, n_concepts, query)

    # =========================================================================
    # CMR WITH JOINT TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 4: CMR with Joint Training")
    print("=" * 60)
    print("Uses DeterministicInference for both training and evaluation")

    cmr_loss = CMRBlendedLoss(task_names=['xor'])
    optim_kwargs_cmr = {'lr': 0.01}

    model_cmr = ConceptMemoryReasoner(
        input_size=n_features,
        annotations=annotations,
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,
        variable_distributions=variable_distributions,
        task_names=['xor'],
        n_rules=10,
        memory_latent_size=100,
        memory_decoder_hidden_layers=1,
        selector_hidden_layers=1,
        hard_roles_at_eval=True,
        inference=DeterministicInference,
        train_inference=DeterministicInference,
        lightning=True,
        loss=cmr_loss,
        rec_weight=0,
        optim_class=optim,
        optim_kwargs=optim_kwargs_cmr,
    )
    print(f"Model type: {type(model_cmr).__name__}")
    print(f"Eval inference: {model_cmr.eval_inference.__class__.__name__}")
    print(f"Training inference: {model_cmr.train_inference.__class__.__name__}")

    trainer_cmr = Trainer(max_epochs=100)
    trainer_cmr.fit(model_cmr, datamodule=datamodule)
    evaluate(model_cmr, datamodule, n_concepts, query)


if __name__ == "__main__":
    main()