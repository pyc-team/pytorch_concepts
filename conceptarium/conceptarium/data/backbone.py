"""
Backbone utilities for feature extraction and embedding precomputation.
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_backbone_embs(
    dataset,
    backbone: nn.Module,
    batch_size: int = 512,
    workers: int = 0,
    show_progress: bool = True
) -> None:
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move backbone to device and set to eval mode
    backbone = backbone.to(device)
    backbone.eval()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order
        num_workers=workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    embeddings_list = []
    
    print("Precomputing embeddings with backbone...")
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader
        for batch in iterator:
            x = batch['x'].to(device) # Extract input data from batch
            embeddings = backbone(x) # Forward pass through backbone
            embeddings_list.append(embeddings.cpu()) # Move back to CPU and store

    all_embeddings = torch.cat(embeddings_list, dim=0) # Concatenate all embeddings
    
    return all_embeddings

def get_backbone_embs(path: str, # path to save/load embeddings
                    dataset,
                    backbone,
                    batch_size,
                    force_recompute=False,  # whether to recompute embeddings even if cached
                    workers=0,
                    show_progress=True):
    # if the path of the embeddings are not precomputed and stored, then compute them and store them
    if not os.path.exists(path) or force_recompute:
        # compute
        embs = compute_backbone_embs(dataset,
                                    backbone,
                                    batch_size=batch_size,
                                    workers=workers,
                                    show_progress=show_progress)
        # save
        print(f"Saving embeddings to {path}")
        torch.save(embs, path)
        print(f"✓ Saved embeddings with shape: {embs.shape}")

    print(f"Loading precomputed embeddings from {path}")
    embs = torch.load(path)
    return embs