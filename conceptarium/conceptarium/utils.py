"""Utility functions for configuration, seeding, and class instantiation.

This module provides helper functions for:
- Setting random seeds across all libraries (re-exported from torch_concepts)
- Configuring runtime environment from Hydra configs
- Dynamic class loading and instantiation
- Managing concept annotations and distributions
"""
import os
import torch
import logging
import torch
from omegaconf import DictConfig, open_dict
from torch_concepts import seed_everything

logger = logging.getLogger(__name__)

from env import DATA_ROOT


def setup_run_env(cfg: DictConfig):
    """Configure runtime environment from Hydra configuration.
    
    Sets up threading, random seeds and matrix multiplication precision.
    
    Args:
        cfg: Hydra DictConfig containing runtime parameters:
            - num_threads: Number of PyTorch threads (default: 1)
            - seed: Random seed for reproducibility
            - matmul_precision: Float32 matmul precision ('highest', 'high', 'medium')
            
    Returns:
        Updated cfg
    """
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    if cfg.get("matmul_precision", None) is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    # set data root
    if not cfg.dataset.get("root"):
        if "name" not in cfg.dataset:
            raise ValueError("If data root is not set, dataset name must be " 
            "specified in cfg.dataset.name to set data root.")
        data_root = os.path.join(DATA_ROOT, cfg.dataset.get("name"))
        with open_dict(cfg):
            cfg.dataset.update(root = data_root)
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
            # FIXME: backbone.output_size might not exist
            input_size = dm.backbone.output_size if dm.backbone else dm.n_features[-1],
            # output_size = sum(dm.concept_metadata.values()),   # check if this is needed
            backbone = dm.backbone if not dm.embs_precomputed else None,
        )
    return cfg
