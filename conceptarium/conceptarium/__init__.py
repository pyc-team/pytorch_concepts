"""
Conceptarium - Training framework for concept-based models.

This module provides PyTorch Lightning-based training infrastructure,
including trainers, experiment utilities, and W&B integration.
"""

from .trainer import Trainer
from .utils import (
    seed_everything,
    setup_run_env,
    clean_empty_configs,
    update_config_from_data,
)
from .wandb import (
    run_from_id,
    checkpoint_from_run,
    model_from_run,
    dataset_from_run,
    iter_runs,
)
from .hydra import target_classname, parse_hyperparams
from .resolvers import register_custom_resolvers

__all__ = [
    # Trainer
    "Trainer",

    # Utilities
    "seed_everything",
    "setup_run_env",
    "clean_empty_configs",
    "update_config_from_data",

    # W&B
    "run_from_id",
    "checkpoint_from_run",
    "model_from_run",
    "dataset_from_run",
    "iter_runs",

    # Hydra
    "target_classname",
    "parse_hyperparams",
    "register_custom_resolvers",
]
