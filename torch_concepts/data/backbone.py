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

def compute_backbone_embs(
    dataset,
    backbone: nn.Module,
    batch_size: int = 512,
    workers: int = 0,
    show_progress: bool = True
) -> torch.Tensor:
    """Extract embeddings from a dataset using a backbone model.
    
    Performs a forward pass through the backbone for the entire dataset and
    returns the concatenated embeddings. Useful for precomputing features
    to avoid repeated backbone computation during training.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'x' key.
        backbone (nn.Module): Feature extraction model (e.g., ResNet encoder).
        batch_size (int, optional): Batch size for processing. Defaults to 512.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        show_progress (bool, optional): Display tqdm progress bar. Defaults to True.
        
    Returns:
        torch.Tensor: Stacked embeddings with shape (n_samples, embedding_dim).
        
    Example:
        >>> from torchvision.models import resnet18
        >>> backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        >>> embeddings = compute_backbone_embs(my_dataset, backbone, batch_size=64)
        >>> embeddings.shape
        torch.Size([10000, 512])
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Store original training state to restore later
    was_training = backbone.training
    
    # Move backbone to device and set to eval mode
    backbone = backbone.to(device)
    backbone.eval()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order
        num_workers=workers,
    )
    
    embeddings_list = []
    
    logger.info("Precomputing embeddings with backbone...")
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader
        for batch in iterator:
            x = batch['x'].to(device) # Extract input data from batch
            embeddings = backbone(x) # Forward pass through backbone
            embeddings_list.append(embeddings.cpu()) # Move back to CPU and store

    all_embeddings = torch.cat(embeddings_list, dim=0) # Concatenate all embeddings
    
    # Restore original training state
    if was_training:
        backbone.train()
    
    return all_embeddings

def get_backbone_embs(path: str,
                    dataset,
                    backbone,
                    batch_size,
                    force_recompute=False,
                    workers=0,
                    show_progress=True):
    """Get backbone embeddings with automatic caching.
    
    Loads embeddings from cache if available, otherwise computes and saves them.
    This dramatically speeds up training by avoiding repeated (pretrained) backbone computation.
    
    Args:
        path (str): File path for saving/loading embeddings (.pt file).
        dataset: Dataset to extract embeddings from.
        backbone: Backbone model for feature extraction.
        batch_size: Batch size for computation.
        force_recompute (bool, optional): Recompute even if cached. Defaults to False.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        show_progress (bool, optional): Show progress bar. Defaults to True.
        
    Returns:
        torch.Tensor: Cached or freshly computed embeddings.
        
    Example:
        >>> embeddings = get_backbone_embs(
        ...     path='cache/mnist_resnet18.pt',
        ...     dataset=train_dataset,
        ...     backbone=my_backbone,
        ...     batch_size=256
        ... )
        Loading precomputed embeddings from cache/mnist_resnet18.pt
    """
    # if the path of the embeddings are not precomputed and stored, then compute them and store them
    if not os.path.exists(path) or force_recompute:
        # compute
        embs = compute_backbone_embs(dataset,
                                    backbone,
                                    batch_size=batch_size,
                                    workers=workers,
                                    show_progress=show_progress)
        # save
        logger.info(f"Saving embeddings to {path}")
        torch.save(embs, path)
        logger.info(f"✓ Saved embeddings with shape: {embs.shape}")

    logger.info(f"Loading precomputed embeddings from {path}")
    embs = torch.load(path)
    return embs
