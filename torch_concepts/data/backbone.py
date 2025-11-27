"""Backbone utilities for feature extraction and embedding precomputation.

Provides functions to extract and cache embeddings from pre-trained backbone
models (e.g., ResNet, ViT) to speed up training of concept-based models.
"""
import os
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def choose_backbone(name: str):
    """Choose a backbone model by name.
    
    Args:
        name (str): Name of the backbone model (e.g., 'resnet18', 'vit_b_16').
        
    Returns:
        tuple: (backbone model, transforms) - The backbone model and its preprocessing transforms.
        
    Raises:
        ValueError: If the backbone name is not recognized.
        
    Example:
        >>> backbone, transforms = choose_backbone('resnet18')
        >>> print(backbone)
        ResNet(...)
    """
    from torchvision.models import (
        resnet18, resnet50, vit_b_16, vit_l_16,
        ResNet18_Weights, ResNet50_Weights, 
        ViT_B_16_Weights, ViT_L_16_Weights
    )

    if name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        transforms = weights.transforms()
        backbone = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    elif name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        transforms = weights.transforms()
        backbone = nn.Sequential(*list(model.children())[:-1])
    elif name == 'vit_b_16':
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        transforms = weights.transforms()
        backbone = nn.Sequential(*list(model.children())[:-1])
    elif name == 'vit_l_16':
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        transforms = weights.transforms()
        backbone = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Backbone '{name}' is not recognized.")
    
    return backbone, transforms

def compute_backbone_embs(
    dataset,
    backbone: str,
    batch_size: int = 512,
    workers: int = 0,
    device: str = None,
    verbose: bool = True
) -> torch.Tensor:
    """Extract embeddings from a dataset using a backbone model.
    
    Performs a forward pass through the backbone for the entire dataset and
    returns the concatenated embeddings. Useful for precomputing features
    to avoid repeated backbone computation during training.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'x' key or 'inputs'.'x' nested key.
        backbone (str): Backbone model name for feature extraction (e.g., 'resnet18').
        batch_size (int, optional): Batch size for processing. Defaults to 512.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'cuda:0', etc.). 
            If None, auto-detects ('cuda' if available, else 'cpu'). Defaults to None.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Stacked embeddings with shape (n_samples, embedding_dim).
        
    Example:
        >>> from torchvision.models import resnet18
        >>> backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        >>> embeddings = compute_backbone_embs(my_dataset, backbone, batch_size=64, device='cuda')
        >>> embeddings.shape
        torch.Size([10000, 512])
    """
    
    # Set device with auto-detection if None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Store original training state to restore later
    #was_training = backbone.training
    
    # Move backbone to device and set to eval mode
    backbone_model, transforms = choose_backbone(backbone)
    backbone_model = backbone_model.to(device)
    backbone_model.eval()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order
        num_workers=workers,
    )
    
    embeddings_list = []
    
    if verbose:
        logger.info("Precomputing embeddings with backbone...")
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
        for batch in iterator:
            # Handle both {'x': tensor} and {'inputs': {'x': tensor}} structures
            if 'inputs' in batch:
                x = batch['inputs']['x'].to(device)
            else:
                x = batch['x'].to(device)
                           
            embeddings = backbone_model(transforms(x)) # Forward pass through backbone
            embeddings_list.append(embeddings.cpu()) # Move back to CPU and store

    all_embeddings = torch.cat(embeddings_list, dim=0) # Concatenate all embeddings
    
    # Restore original training state
    #if was_training:
    #     backbone.train()
    
    return all_embeddings

def get_backbone_embs(path: str,
                    dataset,
                    backbone: str,
                    batch_size,
                    force_recompute=False,
                    workers=0,
                    device=None,
                    verbose=True):
    """Get backbone embeddings with automatic caching.
    
    Loads embeddings from cache if available, otherwise computes and saves them.
    This dramatically speeds up training by avoiding repeated (pretrained) backbone computation.
    
    Args:
        path (str): File path for saving/loading embeddings (.pt file).
        dataset: Dataset to extract embeddings from.
        backbone: Backbone model name for feature extraction.
        batch_size: Batch size for computation.
        force_recompute (bool, optional): Recompute even if cached. Defaults to False.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'cuda:0', etc.).
            If None, auto-detects ('cuda' if available, else 'cpu'). Defaults to None.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Cached or freshly computed embeddings.
        
    Example:
        >>> embeddings = get_backbone_embs(
        ...     path='cache/mnist_resnet18.pt',
        ...     dataset=train_dataset,
        ...     backbone=my_backbone,
        ...     batch_size=256,
        ...     device='cuda'
        ... )
        Loading precomputed embeddings from cache/mnist_resnet18.pt
    """
    # if the path of the embeddings are not precomputed and stored, then compute them and store them
    if not os.path.exists(path) or force_recompute:
        # compute
        embs = compute_backbone_embs(dataset,
                                    backbone=backbone,
                                    batch_size=batch_size,
                                    workers=workers,
                                    device=device,
                                    verbose=verbose)
        # save
        if verbose:
            logger.info(f"Saving embeddings to {path}")
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(embs, path)
        if verbose:
            logger.info(f"✓ Saved embeddings with shape: {embs.shape}")

    if verbose:
        logger.info(f"Loading precomputed embeddings from {path}")
    embs = torch.load(path)
    return embs
