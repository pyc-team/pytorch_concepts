#!/usr/bin/env python
"""Run concept-based model experiments using Hydra configuration."""

import os
import warnings
# Suppress Pydantic warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import logging
logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd

from conceptarium.trainer import Trainer
from conceptarium.hydra import parse_hyperparams
from conceptarium.registry import register_run
from conceptarium.resolvers import register_custom_resolvers
from conceptarium.utils import setup_run_env, maybe_precompute_embeddings, \
    attach_latent_encoder, resolve_graph, update_config_from_data

@hydra.main(config_path="conf", config_name="sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ----------------------------------
    # Setup environment
    # ----------------------------------
    status = "failed"
    cfg = setup_run_env(cfg)

    # ----------------------------------
    # Datamodule
    # 1. Instantiate the datamodule
    # 2. Eventually instantiate the backbone
    # 3. Eventually pre-compute embeddings with backbone
    # 4. Setup the data (split and fit scalers)
    # 5. Update config based on data
    # ----------------------------------
    logger.info("----------------------INIT DATA--------------------------------------")
    datamodule = instantiate(cfg.dataset.datamodule, _convert_="all")
    logger.info(f"Datamodule: {datamodule!r}")
    backbone = instantiate(cfg.dataset.backbone, _convert_="all")
    logger.info(f"Backbone: {backbone}")
    datamodule, backbone = maybe_precompute_embeddings(cfg, datamodule, backbone)
    datamodule.setup('fit')
    cfg = update_config_from_data(cfg, datamodule)

    # ----------------------------------
    # Model
    # 1. Instantiate the loss function
    # 2. Instantiate the metrics
    # 3. Eventually attach the latent encoder to the backbone
    # 4. Instantiate the model
    # ----------------------------------
    logger.info("----------------------INIT MODEL-------------------------------------")
    loss = instantiate(cfg.loss, _convert_="all")
    logger.info(f"Loss: {loss}")

    metrics = instantiate(cfg.metrics, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations
    )
    logger.info(f"Metrics: {metrics}")

    model = instantiate(cfg.model.model_cls, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations,
        graph=resolve_graph(datamodule.graph, datamodule.annotations, cfg.dataset.default_task_names),
        backbone=attach_latent_encoder(cfg, backbone),
        loss=loss,
        metrics=metrics,
    )
    logger.info(f"Model: {model}")
    
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
        status = "success"
    finally:
        trainer.logger.experiment.finish()
        # Log run to CSV registry
        if cfg.get("debug", False):
            logger.warning("Debug mode is ON - skipping registry logging")
        else:
            csv_path = os.path.join(get_original_cwd(), "conceptarium", "runs.csv")
            register_run(os.getcwd(), cfg, status=status, csv_path=csv_path)


if __name__ == "__main__":
    register_custom_resolvers()
    main()