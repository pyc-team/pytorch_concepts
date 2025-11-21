"""Utility functions for configuration, seeding, and class instantiation.

This module provides helper functions for:
- Setting random seeds across all libraries
- Configuring runtime environment from Hydra configs
- Dynamic class loading and instantiation
- Managing concept annotations and distributions
"""

import torch
import numpy as np
import random
import os
import torch
from omegaconf import DictConfig, open_dict

from env import DATA_ROOT


def seed_everything(seed: int):
    """Set random seeds for reproducibility across all libraries.
    
    Sets seeds for Python's random, NumPy, PyTorch CPU and CUDA to ensure
    reproducible results across runs.
    
    Args:
        seed: Integer seed value for random number generators.
        
    Example:
        >>> seed_everything(42)
        Seed set to 42
    """
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_run_env(cfg: DictConfig):
    """Configure runtime environment from Hydra configuration.
    
    Sets up threading, random seeds, matrix multiplication precision, and
    device selection (CUDA/CPU) based on configuration and availability.
    
    Args:
        cfg: Hydra DictConfig containing runtime parameters:
            - num_threads: Number of PyTorch threads (default: 1)
            - seed: Random seed for reproducibility
            - matmul_precision: Float32 matmul precision ('highest', 'high', 'medium')
            
    Returns:
        Updated cfg with 'device' field set to 'cuda' or 'cpu'.
        
    Example:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({'seed': 42, 'num_threads': 4})
        >>> cfg = setup_run_env(cfg)
        >>> print(cfg.device)  # 'cuda' or 'cpu'
    """
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    if cfg.get("matmul_precision", None) is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    with open_dict(cfg): 
        cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    # set DATA_ROOT
    if not cfg.get("DATA_ROOT"):
        with open_dict(cfg):
            cfg.dataset.update(DATA_ROOT=DATA_ROOT)
    return cfg

def clean_empty_configs(cfg: DictConfig) -> DictConfig:
    """Set default None values for missing optional config keys.
    
    Ensures optional configuration sections (causal_discovery, llm, rag) exist
    with None values if not explicitly set, preventing KeyErrors.
    
    Args:
        cfg: Hydra DictConfig to clean.
        
    Returns:
        Updated cfg with default None values for missing keys.
    """
    with open_dict(cfg):
        if not cfg.get('causal_discovery'):
            cfg.update(causal_discovery = None)
        if not cfg.get('llm'):
            cfg.update(llm = None)
        if not cfg.get('rag'):
            cfg.update(rag = None)
    return cfg

def update_config_from_data(cfg: DictConfig, dm) -> DictConfig:
    """Update model configuration from datamodule properties.
    
    Automatically configures model input size, backbone, and embedding settings
    based on the datamodule's dataset properties. This ensures model architecture
    matches the data dimensions.
    
    Args:
        cfg: Hydra DictConfig containing model configuration.
        dm: ConceptDataModule instance with dataset information.
        
    Returns:
        Updated cfg with model.input_size, model.backbone, and 
        model.embs_precomputed set from datamodule.
    """
    with open_dict(cfg):
        cfg.model.update(
            input_size = dm.backbone.output_size if dm.backbone else dm.n_features[-1], # FIXME: backbone.output_size might not exist
            # output_size = sum(dm.concept_metadata.values()),   # check if this is needed
            backbone = dm.backbone,
            embs_precomputed = dm.embs_precomputed
        )
    return cfg
