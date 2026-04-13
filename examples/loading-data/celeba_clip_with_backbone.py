"""
CelebA Concept Bottleneck Model (Low-Level Interface)
======================================================

This example demonstrates how to:
1. Load the CelebA dataset using PyC's dataset utilities
2. Use a pretrained backbone (ResNet50) for feature extraction
3. Build a Concept Bottleneck Model using the low-level API
4. Train the model to predict facial attributes (concepts) and a target task

Key Components:
- CelebADataset: PyC dataset wrapper for CelebA with concept annotations
- Backbone: Pretrained feature extractor (ResNet50, VGG, EfficientNet, DINOv2, etc.)
- LinearLatentToConcept: Maps latent embeddings to concept predictions
- LinearConceptToConcept: Maps concept predictions to task predictions

Dataset: CelebA with 40 binary facial attributes
Task: Predict 'Attractive' attribute from other concept attributes
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data.datasets import CelebADataset, CelebACLIPDataset
from torch_concepts.data.backbone import Backbone
from torch_concepts.nn import LinearLatentToConcept, LinearConceptToConcept


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
    
    # Model configuration
    backbone_name = 'resnet50'  # Options: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0', etc.
    latent_dims = 256           # Dimension of latent space after backbone
    
    # Task configuration - which attribute to predict as the main task
    task_attribute = 'Attractive'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # Load CelebA Dataset
    # =========================================================================
    print("\n1. Loading CelebA dataset...")
    
    # CelebADataset will try to automatically download the raw data if not present
    # in the root directory. If this fails, please manually download the required files
    # ["img_align_celeba.zip", "list_attr_celeba.txt", "list_eval_partition.txt"]
    # and place them in the target root directory.
    # Note: CelebA is a large dataset (~1.4GB for images)
    #dataset = CelebADataset(root='./data/celeba')
    dataset = CelebACLIPDataset(root='./data/celeba')

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
    
    backbone = Backbone(name=backbone_name, device=device)
    
    # Freeze backbone parameters - we only train the CBM layers
    for param in backbone.parameters():
        param.requires_grad = False
    
    # =========================================================================
    # Build Concept Bottleneck Model (Low-Level API)
    # =========================================================================
    print("\n3. Building CBM architecture...")
    
    concept_dims = len(concept_indices) # all binary concepts
    task_dims = 1  # Binary classification
    
    # Latent encoder: reduces backbone features to latent space
    latent_encoder = nn.Sequential(
        nn.Linear(backbone.out_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    
    # Concept encoder: maps latent space to concept predictions
    # Uses PyC's LinearLatentToConcept layer
    concept_encoder = LinearLatentToConcept(
        in_latent=latent_dims,
        out_concepts=concept_dims
    )
    
    # Task predictor: maps concepts to task prediction
    # Uses PyC's LinearConceptToConcept layer
    task_predictor = LinearConceptToConcept(
        in_concepts=concept_dims,
        out_concepts=task_dims
    )
    
    # Combine into a ModuleDict for easy management
    model = nn.ModuleDict({
        'backbone': backbone,
        'latent_encoder': latent_encoder,
        'concept_encoder': concept_encoder,
        'task_predictor': task_predictor,
    }).to(device)
    
    print(f"   Latent dims: {latent_dims}")
    print(f"   Concept dims: {concept_dims}")
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
    loss_fn = nn.BCEWithLogitsLoss()
    
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
            x = batch['inputs']['x'].to(device)  # Images: (B, C, H, W)
            c = batch['concepts']['c'].to(device)  # All concepts: (B, n_concepts)
            
            # Separate concept targets and task target
            c_targets = c[:, concept_indices].float()  # Concept targets
            y_targets = c[:, task_index:task_index+1].float()  # Task target
            
            optimizer.zero_grad()
            
            # Forward pass through CBM
            # 1. Backbone extracts visual features
            features = model['backbone'](x)  # (B, backbone_out_features)
            
            # 2. Latent encoder compresses features
            latent = model['latent_encoder'](features)  # (B, latent_dims)
            
            # 3. Concept encoder predicts concepts
            c_pred = model['concept_encoder'](latent=latent)  # (B, concept_dims)
            
            # 4. Task predictor predicts task from concepts
            y_pred = model['task_predictor'](concepts=c_pred)  # (B, task_dims)
            
            # Compute losses
            concept_loss = loss_fn(c_pred, c_targets)
            task_loss = loss_fn(y_pred, y_targets)
            total_loss = concept_weight * concept_loss + task_weight * task_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_concept_loss += concept_loss.item()
            epoch_task_loss += task_loss.item()
            
            all_concept_preds.append((c_pred.detach() > 0).cpu())
            all_concept_targets.append(c_targets.cpu())
            all_task_preds.append((y_pred.detach() > 0).cpu())
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
        
        concept_acc = accuracy_score(
            (all_concept_targets >= 0.5).float().numpy().flatten(),
            all_concept_preds.numpy().flatten()
        )
        task_acc = accuracy_score(
            (all_task_targets >= 0.5).float().numpy().flatten(),
            all_task_preds.numpy().flatten()
        )
        
        avg_concept_loss = epoch_concept_loss / len(dataloader)
        avg_task_loss = epoch_task_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Concept Loss: {avg_concept_loss:.4f} | Concept Acc: {concept_acc:.4f}")
        print(f"   Task Loss: {avg_task_loss:.4f} | Task Acc: {task_acc:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nThis example demonstrated:")
    print(f"  1. Loading CelebA dataset")
    print(f"  2. Using {backbone_name} backbone for feature extraction")
    print(f"  3. Building a CBM with low-level PyC layers:")
    print(f"     - LinearLatentToConcept: {latent_dims} -> {concept_dims}")
    print(f"     - LinearConceptToConcept: {concept_dims} -> {task_dims}")
    print(f"  4. Training to predict '{task_attribute}' from intermediate concepts")


if __name__ == "__main__":
    main()
