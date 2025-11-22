"""Hydra configuration utilities for extracting metadata and hyperparameters.

This module provides helper functions to parse Hydra/OmegaConf configurations
and extract useful information like class names and hyperparameters for logging.
"""

from omegaconf import DictConfig, OmegaConf

def target_classname(cfg: DictConfig) -> str:
    """Extract the class name from a Hydra configuration's _target_ field.
    
    Args:
        cfg (DictConfig): Configuration with a _target_ field 
            (e.g., "torch_concepts.nn.models.CBM").
    
    Returns:
        str: The class name (e.g., "CBM").
        
    Example:
        >>> cfg = OmegaConf.create({"_target_": "torch_concepts.nn.models.CBM"})
        >>> target_classname(cfg)
        'CBM'
    """
    name = cfg._target_.split(".")[-1]
    return name

def parse_hyperparams(cfg: DictConfig) -> dict[str, any]:
    """Parse configuration to extract key hyperparameters for logging.
    
    Extracts commonly logged hyperparameters like model type, dataset,
    learning rate, seed, and other training configuration. Used primarily
    for W&B logging.
    
    Args:
        cfg (DictConfig): Full Hydra configuration with dataset and model sections.
    
    Returns:
        dict[str, any]: Dictionary containing:
            - dataset: Dataset name (lowercase, without "Dataset" suffix)
            - model: Model class name (lowercase)
            - hidden_size: Hidden layer size (if present in encoder_kwargs)
            - lr: Learning rate
            - seed: Random seed
            - hydra_cfg: Full config as nested dict
            
    Example:
        >>> cfg = OmegaConf.create({
        ...     "dataset": {"_target_": "torch_concepts.data.dataset.MNISTDataset"},
        ...     "model": {"_target_": "torch_concepts.nn.models.CBM",
        ...               "encoder_kwargs": {"hidden_size": 128}},
        ...     "seed": 42
        ... })
        >>> parse_hyperparams(cfg)
        {'dataset': 'mnist', 'model': 'cbm', 'hidden_size': 128, 
         'lr': 0.001, 'seed': 42, 'hydra_cfg': {...}}
    """
    hyperparams = {
        "dataset": target_classname(cfg.dataset).replace("Dataset", "").lower(),
        "model": target_classname(cfg.model).lower(),
        "lr": cfg.model.optim_kwargs.lr,
        "seed": cfg.get("seed"),
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams
