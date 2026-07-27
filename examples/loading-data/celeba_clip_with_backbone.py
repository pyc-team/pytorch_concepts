"""
CelebA Concept Bottleneck Model with CLIP Pseudo-Labels (Low-Level Interface)
==============================================================================

This example demonstrates how to:
1. Load the CelebA dataset using PyC's dataset utilities
2. Use CLIP pseudo-labels (SigLIP2) for concept supervision
3. Use ground-truth CelebA annotations for task supervision only
4. Use a pretrained backbone (ResNet50) for feature extraction
5. Build a Concept Bottleneck Model using the low-level API

Key Components:
- CelebACLIPDataset: CLIP-derived pseudo-labels for concept supervision
- CelebADataset: Ground-truth annotations used only for the task label
- Backbone: Pretrained feature extractor (ResNet50, VGG, EfficientNet, DINOv2, etc.)
- LinearLatentToConcept: Maps latent embeddings to concept predictions
- LinearConceptToConcept: Maps concept predictions to task predictions

Dataset: CelebA with 40 binary facial attributes
Concept supervision: CLIP SigLIP2 pseudo-labels (no human annotations required)
Task supervision: Ground-truth 'Attractive' attribute from CelebA annotations
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch_concepts import seed_everything
from torch_concepts.data.datasets import CelebADataset, CelebACLIPDataset
from torch_concepts.data.backbone import Backbone
from torch_concepts.nn import LinearLatentToConcept, LinearConceptToConcept


class HybridConceptDataset(Dataset):
    """Pairs a CLIP pseudo-label dataset with a ground-truth dataset.

    Returns images and CLIP concept pseudo-labels from ``clip_dataset``,
    and appends the ground-truth concept tensor from ``gt_dataset`` under
    the key ``'gt_concepts'``.  The two datasets must be aligned
    (same images in the same order).
    """

    def __init__(self, clip_dataset: CelebACLIPDataset, gt_dataset: CelebADataset):
        self.clip_dataset = clip_dataset
        self.gt_dataset = gt_dataset

    def __len__(self):
        return len(self.clip_dataset)

    def __getitem__(self, idx):
        clip_sample = self.clip_dataset[idx]
        gt_sample = self.gt_dataset[idx]
        return {
            'inputs':      clip_sample['inputs'],      # images from CLIP dataset
            'concepts':    clip_sample['concepts'],    # CLIP pseudo-labels
            'gt_concepts': gt_sample['concepts'],      # ground-truth annotations
        }


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
    # Load Datasets
    # =========================================================================
    print("\n1. Loading datasets...")

    # CelebADataset / CelebACLIPDataset will try to automatically download the
    # raw data if not present in the root directory.  If this fails, manually
    # place ["img_align_celeba.zip", "list_attr_celeba.txt",
    # "list_eval_partition.txt"] in ./data/celeba/raw/.
    # Note: CelebA is a large dataset (~1.4GB for images).

    # CLIP dataset: provides pseudo-labels used as concept supervision signal.
    clip_dataset = CelebACLIPDataset(root='./data/celeba')

    # Ground-truth dataset: used only to read the task label (Attractive).
    # Both datasets share the same root so images are not re-downloaded.
    gt_dataset = CelebADataset(root='./data/celeba')

    # Concept indices come from the CLIP dataset's vocabulary.
    clip_labels = clip_dataset.annotations[1].labels
    concept_indices = [clip_labels.index(c) for c in clip_labels if c != task_attribute]

    # Task index is looked up in the ground-truth dataset's annotations.
    gt_labels = gt_dataset.annotations[1].labels
    task_index = gt_labels.index(task_attribute)

    print(f"   Dataset size: {len(clip_dataset)} samples")
    print(f"   CLIP concept dims: {len(concept_indices)}")
    print(f"   Task attribute (GT): {task_attribute}")
    
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
    hybrid_dataset = HybridConceptDataset(clip_dataset, gt_dataset)
    subset_indices = list(range(min(max_samples, len(hybrid_dataset))))
    subset = torch.utils.data.Subset(hybrid_dataset, subset_indices)
    
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
            x = batch['inputs']['x'].to(device)           # Images: (B, C, H, W)
            c_clip = batch['concepts']['c'].to(device)    # CLIP pseudo-labels: (B, n_concepts)
            c_gt   = batch['gt_concepts']['c'].to(device) # GT annotations:     (B, n_concepts)

            # Concept supervision: CLIP pseudo-labels (excludes task attribute)
            c_targets = c_clip[:, concept_indices].float()
            # Task supervision: ground-truth annotation only
            y_targets = c_gt[:, task_index:task_index+1].float()
            
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
    print(f"        - Using SigLIP2 pseudo-labels for learning concept layer")
    print(f"     - LinearConceptToConcept: {concept_dims} -> {task_dims}")
    print(f"  4. Training to predict '{task_attribute}' from intermediate concepts")


if __name__ == "__main__":
    main()
