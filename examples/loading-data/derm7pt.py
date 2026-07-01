"""
Derm7pt Concept Bottleneck Model (Low-Level Interface)
======================================================

This example demonstrates how to:
1. Create the Derm7ptDataModule, which loads and splits the Derm7pt dataset
2. Use a pretrained backbone (InceptionV3) for feature extraction
3. Build a Concept Bottleneck Model using the low-level API
4. Train the model to predict skin lesion attributes (concepts) and a target task (diagnosis)
5. Evaluate the model on the held-out test set

Key Components:
- Derm7ptDataModule: Handles data loading, splitting, and batching.
  Internally, it creates a Derm7ptDataset, which is a PyC dataset wrapper for the Derm7pt dataset with concept annotations.
  It also supports precomputing backbone embeddings for faster training.
- Backbone: Pretrained feature extractor (InceptionV3, etc.)
- LinearEmbeddingToConcept: Maps latent embeddings to concept predictions
- LinearConceptToConcept: Maps concept predictions to task predictions

Dataset: Derm7pt with various skin lesion discrete attributes
Task: Predict 'diagnosis' (discrete) attribute from other concept attributes
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data.datamodules.derm7pt import Derm7ptDataModule
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    seed_everything(42)
    
    # Training hyperparameters
    batch_size = 64
    n_epochs = 50
    learning_rate = 0.001
    concept_weight = 10  # Weight for concept loss
    task_weight = 1     # Weight for task loss
    
    # Model configuration
    backbone_name = 'inception_v3'  # Options: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0', 'inception_v3', etc.
    embedding_dims = 256           # Dimension of embeddings space after backbone
    
    # Task configuration - which attribute to predict as the main task
    task_attribute = 'diagnosis'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # =========================================================================
    # Create DataModule and instantiate the backbone
    # =========================================================================
    print("\n1. Creating DataModule and instantiating the backbone...")
    dm = Derm7ptDataModule(
         root='./data/derm7pt',
         batch_size=batch_size,
         backbone=backbone_name,
         precompute_embs=True,  # Precompute embeddings for faster training
     )
    
    # Get the underlying dataset and annotations for concepts
    dataset = dm.dataset
    # Get annotations for concepts
    annotations = dataset.annotations.get_axis_annotation(1)  
    
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Number of concepts: {len(annotations.labels)}")
    print(f"   Task attribute: {task_attribute}")
    
    # Get concept and task indices from annotations
    all_labels = dataset.annotations[1].labels
    concept_indices = [all_labels.index(c) for c in all_labels if c != task_attribute]
    task_index = all_labels.index(task_attribute)

    # =========================================================================
    # Setup datamodule and precompute embeddings if precompute_embs=True
    # =========================================================================
    print("\n2. Setting up datamodule...")
    dm.setup(backbone_device=device)

    print("   Embeddings precomputed/loaded and cached.")
    
    # =========================================================================
    # Build Concept Bottleneck Model (Low-Level API)
    # =========================================================================
    print("\n3. Building CBM architecture...")
    
    concept_output_dims = sum(annotations.cardinalities[i] for i in concept_indices) # all categorical concepts
    task_dims = annotations.cardinalities[task_index]  # Categorical classification
    
    # Latent encoder: reduces backbone features to latent space
    embedding_encoder = nn.Sequential(
        nn.Linear(dm.backbone.out_features, embedding_dims),
        torch.nn.LeakyReLU(),
    )
    
    # Concept encoder: maps latent space to concept predictions
    # Uses PyC's LinearEmbeddingToConcept layer
    concept_encoder = LinearEmbeddingToConcept(
        in_embeddings=embedding_dims,
        out_concepts=concept_output_dims
    )
    
    # Task predictor: maps concepts to task prediction
    # Uses PyC's LinearConceptToConcept layer
    task_predictor = LinearConceptToConcept(
        in_concepts=concept_output_dims,
        out_concepts=task_dims
    )
    
    # Combine into a ModuleDict for easy management
    model = nn.ModuleDict({
        'embedding_encoder': embedding_encoder,
        'concept_encoder': concept_encoder,
        'task_predictor': task_predictor,
    }).to(device)
    
    print(f"   Embedding dims: {embedding_dims}")
    print(f"   Concept dims: {concept_output_dims}")
    print(f"   Task dims: {task_dims}")
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n4. Training CBM...")
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Suitable for multi-class classification of concepts and task
    
    # instantiate the train dataloader
    train_loader = dm.train_dataloader(shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        epoch_concept_loss = 0.0
        epoch_task_loss = 0.0
        concept_correct = 0
        concept_total = 0
        task_correct = 0
        task_total = 0
        epoch_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in progress_bar:
            # Extract inputs and targets from batch
            x = batch['inputs']['x'].to(device)  # Precomputed embeddings (B, backbone_out_features)
            c = batch['concepts']['c'].to(device)  # All concepts: (B, n_concepts)
            batch_size_actual = x.size(0)
            
            # Separate concept targets and task target
            c_targets = c[:, concept_indices].long()  # Concept targets
            y_targets = c[:, task_index].long()  # Task target
            
            optimizer.zero_grad()
            
            # Forward pass through CBM
            # x is already backbone embeddings if precompute_embs=True
            features = x
            
            # 1. Embeddings encoder compresses features extracted by the backbone into a latent space
            embeddings = model['embedding_encoder'](features)  # (B, embedding_dims)
            
            # 2. Concept encoder predicts concepts
            c_pred = model['concept_encoder'](embeddings=embeddings)  # (B, concept_output_dims)
            
            # 3. Task predictor predicts task from concepts
            y_pred = model['task_predictor'](concepts=c_pred)  # (B, task_dims)
            
            #4. Compute concept loss
            offset = 0
            concept_loss = 0.0
            for i, idx in enumerate(concept_indices):
                card = annotations.cardinalities[idx]

                c_pred_i = c_pred[:, offset:offset + card]
                c_pred_classes_i = c_pred_i.argmax(dim=1)
                c_target_i = c_targets[:, i]

                concept_correct += (c_pred_classes_i == c_target_i).sum().item()
                concept_total += c_target_i.numel()
                concept_loss += loss_fn(c_pred_i, c_target_i) 

                offset += card
          
            # concept loss
            concept_loss /= len(concept_indices)

            #5. Compute task loss
            y_hat = y_pred.argmax(dim=1)
            task_correct += (y_hat == y_targets).sum().item()
            task_total += y_targets.numel()
            task_loss = loss_fn(y_pred, y_targets)
            total_loss = concept_weight * concept_loss + task_weight * task_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_concept_loss += concept_loss.item() * batch_size_actual
            epoch_task_loss += task_loss.item() * batch_size_actual
            epoch_samples += batch_size_actual
            

            progress_bar.set_postfix({
                'c_loss': f'{concept_loss.item():.3f}',
                't_loss': f'{task_loss.item():.3f}'
            })
        
        # Compute epoch metrics
        avg_concept_loss = epoch_concept_loss / epoch_samples
        avg_task_loss = epoch_task_loss / epoch_samples
        concept_acc = concept_correct / concept_total
        task_acc = task_correct / task_total
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Concept Loss: {avg_concept_loss:.4f} | Concept Acc: {concept_acc:.4f}")
        print(f"   Task Loss: {avg_task_loss:.4f} | Task Acc: {task_acc:.4f}")
    
    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print("\n5. Evaluating on test set...")

    test_loader = dm.test_dataloader(shuffle=False)

    model.eval()

    concept_correct = 0
    concept_total = 0
    task_correct = 0
    task_total = 0

    test_concept_loss = 0.0
    test_task_loss = 0.0
    test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["inputs"]["x"].to(device)
            c = batch["concepts"]["c"].to(device)
            batch_size_actual = x.size(0)

            c_targets = c[:, concept_indices].long()
            y_targets = c[:, task_index].long()

            # x is already backbone embeddings if precompute_embs=True
            features = x
            embeddings = model["embedding_encoder"](features)
            c_pred = model["concept_encoder"](embeddings=embeddings)
            y_pred = model["task_predictor"](concepts=c_pred)

            # Concept metrics/loss
            offset = 0
            concept_loss = 0.0
            for i, idx in enumerate(concept_indices):
                card = annotations.cardinalities[idx]

                c_pred_i = c_pred[:, offset:offset + card]
                c_pred_classes_i = c_pred_i.argmax(dim=1)
                c_target_i = c_targets[:, i]

                concept_correct += (c_pred_classes_i == c_target_i).sum().item()
                concept_total += c_target_i.numel()
                concept_loss += loss_fn(c_pred_i, c_target_i)
                
                offset += card

            concept_loss /= len(concept_indices)

            # Task metrics/loss
            y_hat = y_pred.argmax(dim=1)
            task_correct += (y_hat == y_targets).sum().item()
            task_total += y_targets.numel()
            task_loss = loss_fn(y_pred, y_targets)
            total_loss = concept_weight * concept_loss + task_weight * task_loss
            
            test_concept_loss += concept_loss.item() * batch_size_actual
            test_task_loss += task_loss.item() * batch_size_actual
            test_samples += batch_size_actual

    test_concept_acc = concept_correct / concept_total
    test_task_acc = task_correct / task_total
    test_concept_loss = test_concept_loss / test_samples
    test_task_loss = test_task_loss / test_samples

    print(f"   Test concept Loss: {test_concept_loss:.4f} | Test concept Acc: {test_concept_acc:.4f}")
    print(f"   Test task Loss: {test_task_loss:.4f} | Test task Acc: {test_task_acc:.4f}")


    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nThis example demonstrated:")
    print(f"  1. Loading Derm7pt datamodule with precomputed embeddings for faster training")
    print(f"  2. Using {backbone_name} backbone for feature extraction")
    print(f"  3. Building a CBM with low-level PyC layers:")
    print(f"     - LinearEmbeddingToConcept: {embedding_dims} -> {concept_output_dims}")
    print(f"     - LinearConceptToConcept: {concept_output_dims} -> {task_dims}")
    print(f"  4. Training to predict '{task_attribute}' from intermediate concepts")
    print(f"  5. Evaluating the trained model on the held-out test set")
    print(f"\nTest set results:")
    print(f"  - Concept Loss: {test_concept_loss:.4f}")
    print(f"  - Task Loss: {test_task_loss:.4f}")
    print(f"  - Concept Accuracy: {test_concept_acc:.4f}")
    print(f"  - Task Accuracy: {test_task_acc:.4f}")


if __name__ == "__main__":
    main()
