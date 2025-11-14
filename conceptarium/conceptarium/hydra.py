from omegaconf import DictConfig, OmegaConf

def target_classname(cfg: DictConfig) -> str:
    name = cfg._target_.split(".")[-1]
    return name

def parse_hyperparams(cfg: DictConfig) -> dict[str, any]:
    hyperparams = {
        "engine": target_classname(cfg.engine)
        .lower(),
        "dataset": target_classname(cfg.dataset)
        .replace("Dataset", "")
        .lower(),
        "model": target_classname(cfg.model)
        .lower(),
        "hidden_size": cfg.model.encoder_kwargs.get("hidden_size", None),
        "lr": cfg.engine.optim_kwargs.lr,
        "seed": cfg.get("seed"),
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams