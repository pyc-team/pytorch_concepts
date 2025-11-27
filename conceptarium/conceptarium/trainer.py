"""PyTorch Lightning Trainer configuration and setup utilities.

This module extends PyTorch Lightning's Trainer class with project-specific
configurations including W&B logging, model checkpointing, and early stopping.
"""

from time import time

from omegaconf import DictConfig
from pytorch_lightning import Trainer as _Trainer_
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

from env import PROJECT_NAME, WANDB_ENTITY
from hydra.core.hydra_config import HydraConfig
from conceptarium.hydra import parse_hyperparams
from wandb.sdk.lib.runid import generate_id     
        
def _get_logger(cfg: DictConfig):
    """Create and configure a W&B logger from Hydra config.
    
    Sets up W&B logging with automatic experiment naming and grouping based
    on dataset, model, and hyperparameters.
    
    Args:
        cfg (DictConfig): Full Hydra configuration containing trainer.logger,
            seed, dataset, model, and hyperparameter settings.
    
    Returns:
        WandbLogger: Configured W&B logger instance.
        
    Raises:
        ValueError: If logger type is not "wandb".
        
    Note:
        Run naming format: "seed{seed}.{timestamp}"
        Group format: "{dataset}.{model}.lr{lr}.{notes}"
    """
    name = f"seed{cfg.get('seed', '')}.{int(time())}"
    group_format = (
        "{dataset}.{model}.lr{lr}"
    )
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.get("notes") is not None:
        group = f"{group}.{cfg.notes}"
    if cfg.trainer.logger == "wandb":
        logger = WandbLogger(
            project=PROJECT_NAME,
            entity=WANDB_ENTITY,
            log_model=True,
            id=generate_id(),
            save_dir=HydraConfig.get().runtime.output_dir,
            name=name,
            group=group,
        )
    else:
        raise ValueError(f"Unknown logger {cfg.trainer.logger}")
    return logger


class Trainer(_Trainer_):
    """Extended PyTorch Lightning Trainer with project-specific defaults.
    
    Automatically configures:
    - Model checkpointing (saves best model based on monitored metric)
    - Early stopping (if patience is specified)
    - Learning rate monitoring
    - W&B logging (if logger is specified)
    - Device accelerator from config
    
    Args:
        cfg (DictConfig): Hydra configuration containing trainer settings:
            - trainer.monitor: Metric to monitor for checkpointing/early stopping
            - trainer.patience: Early stopping patience (epochs)
            - trainer.logger: Logger type ("wandb" or None for DummyLogger)
            - Other pytorch_lightning.Trainer arguments
            
    Example:
        >>> cfg = OmegaConf.create({
        ...     "trainer": {
        ...         "max_epochs": 100,
        ...         "monitor": "val_loss",
        ...         "patience": 10,
        ...         "logger": "wandb"
        ...     },
        ...     "seed": 42,
        ...     "dataset": {"_target_": "..."},
        ...     "model": {"_target_": "..."}
        ... })
        >>> trainer = Trainer(cfg)
        >>> trainer.fit(model, datamodule)
    """
    def __init__(self, cfg: DictConfig):
        callbacks = []
        if cfg.trainer.get("monitor", None) is not None:
            if cfg.trainer.get("patience", None) is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=cfg.trainer.monitor,
                        patience=cfg.trainer.patience,
                    )
                )
            callbacks.append(
                ModelCheckpoint(
                    dirpath="checkpoints",
                    every_n_epochs=None,
                    monitor=cfg.trainer.monitor,
                    save_top_k=1,
                    mode="min",
                    save_last=True,
                    save_weights_only=False,
                )
            )
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            )
        )

        # logger selection and setup
        if cfg.trainer.get("logger") is not None:
            logger = _get_logger(cfg)
        else:
            logger = DummyLogger()

        trainer_kwargs = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["monitor", "patience", "logger"]
        }
        super().__init__(
            callbacks=callbacks,
            logger=logger,
            **trainer_kwargs,
        )
