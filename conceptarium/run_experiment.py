#!/usr/bin/env python
"""Run concept-based model experiments using Hydra configuration."""

import warnings
# Suppress Pydantic warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import logging
logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from conceptarium.trainer import Trainer
from conceptarium.hydra import parse_hyperparams
from conceptarium.resolvers import register_custom_resolvers
from conceptarium.utils import setup_run_env, clean_empty_configs, update_config_from_data

@hydra.main(config_path="conf", config_name="sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ----------------------------------
    # Setup environment
    # ----------------------------------
    cfg = setup_run_env(cfg)
    cfg = clean_empty_configs(cfg)

    # ----------------------------------
    # Dataset
    # 
    # 1. Instantiate the datamodule
    # 2. Setup the data (preprocess with backbone, split, fit scalers)
    # 3. Update config based on data
    # ----------------------------------
    logger.info("----------------------INIT DATA--------------------------------------")
    datamodule = instantiate(cfg.dataset, _convert_="all")
    datamodule.setup('fit', verbose=True)
    cfg = update_config_from_data(cfg, datamodule)

    # ----------------------------------
    # Model
    # 1. Instantiate the loss function
    # 2. Instantiate the model
    # ----------------------------------
    logger.info("----------------------INIT MODEL-------------------------------------")
    loss = instantiate(cfg.loss, annotations=datamodule.annotations, _convert_="all")
    logger.info(loss)
    metrics = instantiate(cfg.metrics, annotations=datamodule.annotations, _convert_="all")
    logger.info(metrics)
    model = instantiate(cfg.model, annotations=datamodule.annotations, loss=loss, metrics=metrics, _convert_="all")
    
    logger.info("----------------------BEGIN TRAINING---------------------------------")
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        # ----------------------------------
        # Train
        trainer.fit(model, datamodule=datamodule)
        # ----------------------------------
        # TODO: implement finetuning
        # if cfg.get("finetune") is not None:
        #     trainer = maybe_finetune_model(trainer, cfg.finetune)
        # ----------------------------------
        # Test
        trainer.test(datamodule=datamodule)
        # ----------------------------------
        
        trainer.logger.finalize("success")
    finally:
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    register_custom_resolvers()
    main()