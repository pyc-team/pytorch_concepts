"""
CEBaB Concept Bottleneck Model (Low-Level Interface)
======================================================

This example demonstrates how to:
1. Load the CEBaB dataset using PyC's dataset utilities
2. Use a pretrained backbone (bert-base-uncased) for feature extraction
3. Build a Concept Bottleneck Model using the low-level API
4. Train the model to predict restaurant-related variables (concepts) and a target task (review) from a textual input (description).

Key Components:
- CEBaBDataset: PyC dataset wrapper for CEBaB with concept annotations
- Backbone: Pretrained feature extractor (bert-base-uncased, etc.)
- LinearEmbeddingToConcept: Maps latent embeddings to concept predictions
- LinearConceptToConcept: Maps concept predictions to task predictions

Dataset: CEBaB with 4 categorical (if concepts_type = 'discrete') or continuous (if concepts_type = 'continuous') restaurant-related attributes
Task: Predict categorical 'review' (if concepts_type = 'discrete') or continuous 'review' (if concepts_type = 'continuous') from other concept attributes
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data import CEBaBDataset
from torch_concepts.data.backbone import Backbone
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    seed_everything(42)
    
    # Training hyperparameters
    batch_size = 16
    n_epochs = 100
    learning_rate = 0.01
    concept_weight = 10  # Weight for concept loss
    task_weight = 1     # Weight for task loss
    ## specific for CEBaB ##
    concepts_type = 'discrete'  # 'discrete' or 'continuous'
    input_type = 'text'  # 'text' or 'image'
    
    # Model configuration
    backbone_name = 'bert-base-uncased'  # Options: 'bert-base-uncased', etc. 
    embedding_dims = 256           # Dimension of embedding space after backbone
    
    # Task configuration - which attribute to predict as the main task
    task_attribute = 'review'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # Load CEBaB Dataset
    # =========================================================================
    print("\n1. Loading CEBaB dataset...")
    
    # CEBaBDataset will automatically download the raw data from Hugging Face Datasets
    # and place them in the target root directory, if not already present.
    dataset = CEBaBDataset(root='./data/CEBaB', concepts_type=concepts_type)

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
    # Initialize Backbone for Feature Extraction
    # =========================================================================
    print(f"\n2. Loading backbone: {backbone_name}...")
    
    backbone = Backbone(name=backbone_name, device=device, input_type=input_type)
    
    # Freeze backbone parameters - we only train the CBM layers
    for param in backbone.parameters():
        param.requires_grad = False
    
    # =========================================================================
    # Build Concept Bottleneck Model (Low-Level API)
    # =========================================================================
    print("\n3. Building CBM architecture...")

    print("   Using discrete concepts and task. Loss: CrossEntropyLoss")
    concept_output_dims = sum(annotations.cardinalities[i] for i in concept_indices)
    task_dims = annotations.cardinalities[task_index]
    # specify loss functions for discrete concepts and task
    loss_fn = nn.CrossEntropyLoss()


    # Embedding encoder: reduces backbone features to embedding space
    embedding_encoder = nn.Sequential(
        nn.Linear(backbone.out_features, embedding_dims),
        torch.nn.LeakyReLU(),
    )
    
    # Concept encoder: maps embedding space to concept predictions
    # Uses PyC's LinearEmbeddingToConcept layer
    # output: logits for each concept
    concept_encoder = LinearEmbeddingToConcept(
        in_embeddings=embedding_dims,
        out_concepts=concept_output_dims
    )
    
    # Task predictor: maps concepts to task prediction
    # Uses PyC's LinearConceptToConcept layer
    # output: logits for task prediction
    task_predictor = LinearConceptToConcept(
        in_concepts=concept_output_dims,
        out_concepts=task_dims
    )
    
    # Combine into a ModuleDict for easy management
    model = nn.ModuleDict({
        'backbone': backbone,
        'embedding_encoder': embedding_encoder,
        'concept_encoder': concept_encoder,
        'task_predictor': task_predictor,
    }).to(device)
    
    print(f"   Embedding dims: {embedding_dims}")
    print(f"   Number of concepts: {len(concept_indices)}")
    print(f"   Concept output dims: {concept_output_dims}")
    print(f"   Task dims: {task_dims}")
    
    # =========================================================================
    # Create DataLoader
    # =========================================================================
    print("\n4. Creating DataLoader...")
    
    # Use a smaller subset for this example to speed up training
    max_samples = 100
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging; increase for production
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"   Subset size: {len(subset)} samples")
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n5. Training CBM...")
    
    # Only optimize parameters that require gradients (excludes frozen backbone)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    
    model.train()
    for epoch in range(n_epochs):
        epoch_concept_loss = 0.0
        epoch_task_loss = 0.0
        all_concept_preds = []
        all_concept_targets = []
        all_task_preds = []
        all_task_targets = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in progress_bar:
            # Extract inputs and targets from batch
            x = batch['inputs']['x']  # Text
            c = batch['concepts']['c'].to(device)  # All concepts: (B, n_concepts)
            
            # Separate concept targets and task target
            c_targets = c[:, concept_indices].long()
            y_targets = c[:, task_index].long()

            optimizer.zero_grad()
            
            # Forward pass through CBM
            # 1. Backbone extracts textual features
            features = model['backbone'](x)  # (B, backbone_out_features)
            
            # 2. Embedding encoder compresses features
            embeddings = model['embedding_encoder'](features)  # (B, embedding_dims)
            
            # 3. Concept encoder predicts concepts
            c_pred = model['concept_encoder'](embeddings=embeddings)  # (B, concept_dims)
            
            
            concept_loss = 0
            c_probs = []
            offset = 0

            for i, idx in enumerate(concept_indices):
                card = annotations.cardinalities[idx]
                c_pred_i = c_pred[:, offset:offset + card]

                c_probs.append(torch.softmax(c_pred_i, dim=1))

                c_target_i = c_targets[:, i]
                concept_loss += loss_fn(c_pred_i, c_target_i)

                offset += card

            # concept loss
            concept_loss /= len(concept_indices)

            # 4. Task predictor predicts task from concept features.
            c_for_task = torch.cat(c_probs, dim=1)
            y_pred = model['task_predictor'](concepts=c_for_task)

            # task loss
            task_loss = loss_fn(y_pred, y_targets)
            total_loss = concept_weight * concept_loss + task_weight * task_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_concept_loss += concept_loss.item()
            epoch_task_loss += task_loss.item()
            
            all_concept_preds.append(c_pred.detach().cpu())
            all_concept_targets.append(c_targets.cpu())
            all_task_preds.append(y_pred.detach().cpu())
            all_task_targets.append(y_targets.cpu())
            
            progress_bar.set_postfix({
                'c_loss': f'{concept_loss.item():.3f}',
                't_loss': f'{task_loss.item():.3f}'
            })
        
        # Compute epoch metrics
        all_concept_preds = torch.cat(all_concept_preds, dim=0)
        all_concept_targets = torch.cat(all_concept_targets, dim=0)
        all_task_preds = torch.cat(all_task_preds, dim=0)
        all_task_targets = torch.cat(all_task_targets, dim=0)

        avg_concept_loss = epoch_concept_loss / len(dataloader)
        avg_task_loss = epoch_task_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        c_pred_classes = []
        offset = 0
        for i, idx in enumerate(concept_indices):
            card = annotations.cardinalities[idx]
            c_pred_i = all_concept_preds[:, offset:offset+card]
            c_pred_classes.append(c_pred_i.argmax(dim=1))
            offset += card
        c_pred_classes = torch.stack(c_pred_classes, dim=1)  # shape: (B, n_concepts)

        c_target_classes = all_concept_targets.long()

        concept_acc = accuracy_score(
                c_target_classes.flatten().numpy(),
                c_pred_classes.flatten().numpy()
            )

        y_pred_classes = all_task_preds.argmax(dim=1).squeeze()
        y_target_classes = all_task_targets.long().squeeze()
        task_acc = accuracy_score(
            y_target_classes.numpy(),
            y_pred_classes.numpy()
        )

        print(f"   Concept Loss: {avg_concept_loss:.4f} | Concept Acc: {concept_acc:.4f}")
        print(f"   Task Loss: {avg_task_loss:.4f} | Task Acc: {task_acc:.4f}")

    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nThis example demonstrated:")
    print(f"  1. Loading CEBaB dataset using discrete concepts")
    print(f"  2. Using {backbone_name} backbone for feature extraction")
    print(f"  3. Building a CBM with low-level PyC layers:")
    print(f"     - LinearEmbeddingToConcept: {embedding_dims} -> {concept_output_dims}")
    print(f"     - LinearConceptToConcept: {concept_output_dims} -> {task_dims}")
    print(f"  4. Training to predict '{task_attribute}' from intermediate concepts")


if __name__ == "__main__":
    main()
