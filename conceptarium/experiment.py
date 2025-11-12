# Configure warnings before importing any third-party libraries
import conceptarium.warnings_config  # noqa: F401 - suppress WandB/Pydantic warnings

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
    datamodule = instantiate(cfg.dataset, _convert_="all")
    datamodule.setup('fit')
    cfg = update_config_from_data(cfg, datamodule)

    # ----------------------------------
    # Model
    # 
    # 1. Instantiate the model
    # ----------------------------------
    model = instantiate(cfg.model, _convert_="all", 
                        _partial_=True)(annotations=datamodule.annotations,
                                        graph=datamodule.graph)

    # ----------------------------------
    # Engine
    #
    # 1. Instantiate the engine, passing the model as argument
    # ----------------------------------
    engine = instantiate(cfg.engine, _convert_="all", 
                         _partial_=True)(model=model)

    print("-------------------------------------------------------")
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        # maybe_set_summary_metrics(trainer.logger, engine)
        # ----------------------------------
        # Train
        trainer.fit(engine, datamodule=datamodule)
        # ----------------------------------
        # Finetune
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