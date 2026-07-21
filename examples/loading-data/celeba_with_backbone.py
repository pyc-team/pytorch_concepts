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
from pytorch_lightning import Trainer

from torch_concepts import seed_everything, ImageBackbone
from torch_concepts.data import CelebADataModule
from torch_concepts.nn import MLP, ConceptBottleneckModel, ConceptLoss


def main():
    seed_everything(42)
    
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
    dm = CelebADataModule(
        root='./data/celeba', 
        max_samples=10000,
        splitter=None,
        val_size=0.1,
        test_size=0.2,
        batch_size=512
    )
    print(f"   Dataset size: {dm.n_samples} samples")
    print(f"   Image shape: {dm.n_features}")
    print(f"   Number of concepts: {dm.n_concepts}")
    
    # =========================================================================
    # Initialize Backbone for Feature Extraction
    # =========================================================================
    print(f"\n2. Loading backbone...")
    
    backbone = ImageBackbone(name='resnet18', device=device, freeze=True)
    print(backbone)
    
    # =========================================================================
    # Build a simple CBM
    # =========================================================================
    print("\n3. Building CBM architecture...")
    
    model = ConceptBottleneckModel(
        input_size=dm.n_features,
        annotations=dm.annotations,
        task_names=['Attractive'],

        backbone=torch.nn.Sequential(
            backbone,
            MLP(backbone.out_features, 64)
        ),
        latent_size=64,

        lightning=True,
        loss=ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.01},
    )
    trainer = Trainer(max_epochs=50, logger=False)
    trainer.fit(model, datamodule=dm)

    trainer.test(ckpt_path='best', datamodule=dm)


if __name__ == "__main__":
    main()
